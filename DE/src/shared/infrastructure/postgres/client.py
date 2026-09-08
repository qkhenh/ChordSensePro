"""Async SQLAlchemy session factory for PostgreSQL with get_session() context manager (auto commit/rollback)."""

from __future__ import annotations
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
# pyrefly: ignore [missing-import]
from src.shared.infrastructure.settings.config import get_settings

_engine = None
_session_factory = None

def _get_session_factory() -> async_sessionmaker:
    """Lazy init — engine is only created on first call, not at import time."""
    global _engine, _session_factory
    if _session_factory is None:
        url = get_settings().postgres.url
        _engine = create_async_engine(url, pool_size=5, max_overflow=10, echo=False)
        _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
    return _session_factory

@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager — used in repositories and API handlers.
    Example:
        async with get_session() as session:
            result = await session.execute(...)
    """
    async with _get_session_factory()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


