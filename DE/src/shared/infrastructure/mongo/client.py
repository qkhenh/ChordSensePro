"""
shared/infrastructure/mongo/client.py

TODO: Bạn sẽ viết async MongoDB client dùng Motor

Bạn cần viết:
- Tạo AsyncIOMotorClient từ settings.mongo.uri
- Lấy database theo settings.mongo.db_name
- Singleton get_mongo_db() trả về database instance
"""
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
    """Trả về database instance — dùng trong mọi MongoDB repository.
    Ví dụ:
        db = get_mongo_db()
        await db["raw_audio"].insert_one({...})
    """
    return get_mongo_client()[get_settings().mongo.db_name]

async def close_mongo() -> None:
    """Gọi khi app shutdown để đóng connection pool."""
    global _client
    if _client is not None:
        _client.close()
        _client = None