"""
shared/infrastructure/postgres/client.py

TODO: Bạn sẽ viết async PostgreSQL client dùng SQLAlchemy 2.0

Bạn cần viết:
- Tạo async engine từ settings.postgres.url
- Tạo AsyncSession factory
- Async context manager get_session() để dùng trong các repository
"""

from __future__ import annotations
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
# pyrefly: ignore [missing-import]
from src.shared.infrastructure.settings.config import get_settings

def _make_engine():
    url = get_settings().postgres.url
    return create_async_engine(
        url,
        pool_size=5,
        max_overflow=10,
        echo=False,   # True để debug SQL
    )
    
_engine = _make_engine()
_session_factory = async_sessionmaker(_engine, expire_on_commit=False)

@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager — dùng trong repository và API handlers.
    Ví dụ:
        async with get_session() as session:
            result = await session.execute(...)
    """
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


