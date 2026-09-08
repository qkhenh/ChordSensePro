"""Motor async MongoDB client singleton with get_mongo_db() and close_mongo() for shutdown."""

from __future__ import annotations
# pyrefly: ignore [missing-import]
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
# pyrefly: ignore [missing-import]
from src.shared.infrastructure.settings.config import get_settings

_client: AsyncIOMotorClient | None = None

def get_mongo_client() -> AsyncIOMotorClient:
    """Singleton Motor client."""
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(get_settings().mongo.uri)
    return _client

def get_mongo_db() -> AsyncIOMotorDatabase:
    """Return the database instance — used by all MongoDB repositories.
    Example:
        db = get_mongo_db()
        await db["raw_audio"].insert_one({...})
    """
    return get_mongo_client()[get_settings().mongo.db_name]

async def close_mongo() -> None:
    """Call on app shutdown to close the connection pool."""
    global _client
    if _client is not None:
        _client.close()
        _client = None