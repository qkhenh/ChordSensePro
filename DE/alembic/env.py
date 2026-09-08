"""Alembic env.py — auto-generates migrations from ORM metadata.

Imports all ORM models so Base.metadata contains every table.
Run: alembic revision --autogenerate -m "init"
     alembic upgrade head
"""
import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import create_async_engine

from src.shared.infrastructure.postgres.orm_base import Base
from src.shared.infrastructure.settings.config import get_settings

# ── Import all ORM models so they register with Base.metadata ──
from src.song_analysis.infrastructure.orm.song_analysis_orm import SongAnalysisORM       # noqa: F401
from src.user_analytics.infrastructure.orm.chord_attempt_orm import ChordAttemptORM       # noqa: F401
from src.user_analytics.infrastructure.orm.chord_mastery_orm import ChordMasteryORM       # noqa: F401
from src.data_ingest.infrastructure.orm.crawl_queue_orm import CrawlQueueORM              # noqa: F401


config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode — generates SQL without DB connection."""
    url = get_settings().postgres.url
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in 'online' mode with async engine."""
    url = get_settings().postgres.url
    connectable = create_async_engine(url, poolclass=pool.NullPool)
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
