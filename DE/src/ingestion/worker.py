"""Celery worker entry point.

    rtk uv run celery -A src.ingestion.worker worker --pool=solo --loglevel=INFO

Needs the same environment as the API (DATABASE_URL/POSTGRES_*, AUTH_JWT_SECRET, INGEST_*).
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from src.api.wiring import build_container, build_ingestion
from src.identity.settings import AuthSettings
from src.ingestion.application.executor import JobExecutor
from src.ingestion.infrastructure.celery_app import configure, make_celery
from src.ingestion.infrastructure.events_redis import RedisJobEvents
from src.ingestion.settings import IngestSettings
from src.persistence import database_url


@asynccontextmanager
async def open_executor() -> AsyncIterator[JobExecutor]:
    """Services for one task. No pooling: the next task runs in a different event loop."""
    engine = create_async_engine(database_url(), poolclass=NullPool)
    ingestion = build_ingestion(
        IngestSettings(job_runner="celery"), engine, build_container(AuthSettings(), engine)
    )
    try:
        yield ingestion.executor
    finally:
        if isinstance(ingestion.events, RedisJobEvents):
            await ingestion.events.aclose()
        await engine.dispose()


_settings = IngestSettings()
app = make_celery(_settings.redis_url, job_timeout_s=_settings.job_timeout_s)
configure(open_executor)
