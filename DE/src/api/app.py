"""Application factory: ``uvicorn src.api.app:create_app --factory``."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import create_async_engine

from src.api.container import Container, IngestionContainer
from src.api.errors import install_error_handlers
from src.api.limits import BodySizeLimit
from src.api.middleware import REQUEST_ID_HEADER, RequestContextMiddleware
from src.api.routers import analyze, auth, charts, demo, jobs_ws, users
from src.api.settings import ApiSettings
from src.api.wiring import build_container, build_ingestion
from src.identity.settings import AuthSettings
from src.ingestion.application.cleanup import TempSweeper
from src.ingestion.settings import IngestSettings
from src.persistence import database_url

# The multipart envelope (boundaries, headers) rides on top of the file itself.
_ENVELOPE_BYTES = 64 * 1024


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the PostgreSQL-backed services unless they were injected (tests)."""
    engine = None
    if not hasattr(app.state, "container"):
        engine = create_async_engine(database_url(), pool_pre_ping=True)
        app.state.engine = engine
        app.state.container = build_container(AuthSettings(), engine)
        app.state.ingestion = build_ingestion(IngestSettings(), engine, app.state.container)
    ingestion = getattr(app.state, "ingestion", None)
    if ingestion is None:  # an identity-only app (tests): nothing to recover or sweep
        yield
        return
    # In-process, this process started every job there is, so any unfinished one is dead.
    # With Celery workers some may still be running: wait for the whole-job deadline.
    own_every_job = ingestion.settings.job_runner == "inprocess"
    await ingestion.executor.recover_orphans(0 if own_every_job else None)
    sweeper = TempSweeper(
        ingestion.settings.data_dir / "tmp", ingestion.settings.temp_ttl_s, ingestion.jobs
    )
    await sweeper.sweep()
    app.state.sweeper = asyncio.create_task(
        sweeper.run_forever(max(ingestion.settings.temp_ttl_s / 4, 1))
    )
    app.state.recovery = asyncio.create_task(
        ingestion.executor.run_recovery_forever(max(ingestion.settings.job_deadline_s / 4, 1))
    )
    try:
        yield
    finally:
        for background in (app.state.sweeper, app.state.recovery):
            background.cancel()
            with suppress(asyncio.CancelledError):
                await background
        await ingestion.runner.drain()
        if engine is not None:
            await engine.dispose()


def create_app(
    container: Container | None = None,
    settings: ApiSettings | None = None,
    ingestion: IngestionContainer | None = None,
) -> FastAPI:
    settings = settings or ApiSettings()
    app = FastAPI(title="ChordSense Pro API", version="0.1.0", lifespan=_lifespan)
    if container is not None:
        app.state.container = container
    if ingestion is not None:
        app.state.ingestion = ingestion
    install_error_handlers(app)

    upload_limit = (ingestion.settings if ingestion else IngestSettings()).max_upload_bytes
    # Innermost, so its 413 still gets a request id and CORS headers on the way out.
    app.add_middleware(
        BodySizeLimit, limits={"/api/v1/analyze/file": upload_limit + _ENVELOPE_BYTES}
    )
    # Added last = outermost, so CORS headers and preflights cover every response.
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "If-None-Match", REQUEST_ID_HEADER],
        expose_headers=[REQUEST_ID_HEADER, "Retry-After", "Content-Range", "Accept-Ranges", "ETag"],
    )

    @app.get("/healthz", tags=["ops"])
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(auth.router, prefix="/api/v1")
    app.include_router(users.router, prefix="/api/v1")
    app.include_router(analyze.router, prefix="/api/v1")
    app.include_router(charts.router, prefix="/api/v1")
    app.include_router(jobs_ws.router, prefix="/api/v1")
    app.include_router(demo.router, prefix="/api/v1")
    return app
