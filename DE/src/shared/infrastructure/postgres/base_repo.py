"""Generic async SQLAlchemy repository base class with save, get_by_id, delete, and list operations."""
from __future__ import annotations
from typing import Generic, TypeVar
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

T = TypeVar("T")

class BaseRepository(Generic[T]):
    """Generic async repository. Subclass and set `model` to the ORM class.

    Example:
        class SongRepository(BaseRepository[SongAnalysisORM]):
            model = SongAnalysisORM
    """
    model: type[T]
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        
    async def get_by_id(self, id: str) -> T | None:
        result = await self.session.get(self.model, id)
        return result
    
    async def save(self, entity: T) -> T:
        self.session.add(entity)
        await self.session.flush()  # flush to get generated fields, not committed yet
        return entity
    
    async def delete_by_id(self, id: str) -> None:
        await self.session.execute(
            delete(self.model).where(self.model.id == id)  # type: ignore[attr-defined]
        )
        
    async def list_all(self) -> list[T]:
        result = await self.session.execute(select(self.model))
        return list(result.scalars().all())
