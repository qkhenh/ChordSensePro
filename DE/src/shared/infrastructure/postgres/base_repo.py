from __future__ import annotations
from typing import Generic, TypeVar
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

T = TypeVar("T")

class BaseRepository(Generic[T]):
    """Generic async repository — kế thừa để dùng cho từng Entity.
    Ví dụ dùng:
        class SongRepository(BaseRepository[SongORM]):
            model = SongORM
    """
    model: type[T]
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        
    async def get_by_id(self, id: str) -> T | None:
        result = await self.session.get(self.model, id)
        return result
    
    async def save(self, entity: T) -> T:
        self.session.add(entity)
        await self.session.flush()  # flush để lấy generated fields, chưa commit
        return entity
    
    async def delete_by_id(self, id: str) -> None:
        await self.session.execute(
            delete(self.model).where(self.model.id == id)  # type: ignore[attr-defined]
        )
        
    async def list_all(self) -> list[T]:
        result = await self.session.execute(select(self.model))
        return list(result.scalars().all())
