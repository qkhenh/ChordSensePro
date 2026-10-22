"""Alembic environment (async, asyncpg). All objects live in the ``app`` schema."""

import asyncio

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from src.persistence import database_url
from src.persistence.tables import SCHEMA, metadata

config = context.config


def _url() -> str:
    return config.get_main_option("sqlalchemy.url") or database_url()


def _include_name(name, type_, parent_names):
    return type_ != "schema" or name == SCHEMA


def _configure(connection=None, url=None) -> None:
    context.configure(
        connection=connection,
        url=url,
        target_metadata=metadata,
        include_schemas=True,
        include_name=_include_name,
        version_table_schema=SCHEMA,
        compare_type=True,
        literal_binds=connection is None,
        dialect_opts={"paramstyle": "named"} if connection is None else None,
    )


def run_migrations_offline() -> None:
    _configure(url=_url())
    with context.begin_transaction():
        context.run_migrations()


def _run_sync(connection) -> None:
    _configure(connection=connection)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = create_async_engine(_url(), poolclass=NullPool)
    try:
        async with engine.connect() as connection:
            await connection.run_sync(_run_sync)
    finally:
        await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
