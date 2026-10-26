"""Application factory: ``uvicorn src.api.app:create_app --factory``."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import create_async_engine

from src.api.container import Container
from src.api.errors import install_error_handlers
from src.api.middleware import REQUEST_ID_HEADER, RequestContextMiddleware
from src.api.routers import auth, users
from src.api.settings import ApiSettings
from src.api.wiring import build_container
from src.identity.settings import AuthSettings
from src.persistence import database_url


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the PostgreSQL-backed container unless one was injected (tests)."""
    if hasattr(app.state, "container"):
        yield
        return
    engine = create_async_engine(database_url(), pool_pre_ping=True)
    app.state.engine = engine
    app.state.container = build_container(AuthSettings(), engine)
    try:
        yield
    finally:
        await engine.dispose()


def create_app(container: Container | None = None, settings: ApiSettings | None = None) -> FastAPI:
    settings = settings or ApiSettings()
    app = FastAPI(title="ChordSense Pro API", version="0.1.0", lifespan=_lifespan)
    if container is not None:
        app.state.container = container
    install_error_handlers(app)

    # Added last = outermost, so CORS headers and preflights cover every response.
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE"],
        allow_headers=["Authorization", "Content-Type", REQUEST_ID_HEADER],
        expose_headers=[REQUEST_ID_HEADER, "Retry-After"],
    )

    @app.get("/healthz", tags=["ops"])
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(auth.router, prefix="/api/v1")
    app.include_router(users.router, prefix="/api/v1")
    return app
