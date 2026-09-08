"""Repository for crawl_queue table."""
from __future__ import annotations
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.shared.infrastructure.postgres.base_repo import BaseRepository
from src.data_ingest.infrastructure.orm.crawl_queue_orm import CrawlQueueORM
from src.data_ingest.domain.models.crawl_queue_item import CrawlQueueItem, CrawlStatus


class CrawlQueueRepository(BaseRepository[CrawlQueueORM]):
    """Async repository for crawl_queue table."""
    model = CrawlQueueORM

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def get_pending(self, limit: int = 10) -> list[CrawlQueueItem]:
        """Fetch N pending items ordered by priority DESC, created_at ASC.

        Used by dag_crawl_batch to pick URLs to process each run.
        """
        stmt = (
            select(CrawlQueueORM)
            .where(CrawlQueueORM.status == CrawlStatus.PENDING.value)
            .order_by(CrawlQueueORM.priority.desc(), CrawlQueueORM.created_at.asc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return [row.to_domain() for row in result.scalars()]

    async def add_url(self, source_url: str, priority: int = 0) -> CrawlQueueItem | None:
        """Insert a new URL into the queue. Returns None if URL already exists (dedup).

        Uses INSERT ... ON CONFLICT DO NOTHING pattern via try/except.
        """
        from sqlalchemy.exc import IntegrityError
        item = CrawlQueueItem(source_url=source_url, priority=priority)
        orm  = CrawlQueueORM.from_domain(item)
        try:
            self.session.add(orm)
            await self.session.flush()
            return item
        except IntegrityError:
            await self.session.rollback()
            return None   # URL already in queue — skip silently

    async def update_status(
        self,
        item_id: str,
        status: CrawlStatus,
        error_message: str | None = None,
    ) -> None:
        """Update status (and optional error_message) for one queue item."""
        from datetime import datetime, timezone
        values: dict = {"status": status.value}
        if status in (CrawlStatus.DONE, CrawlStatus.FAILED, CrawlStatus.SKIPPED):
            values["processed_at"] = datetime.now(timezone.utc)
        if error_message is not None:
            values["error_message"] = error_message

        await self.session.execute(
            update(CrawlQueueORM)
            .where(CrawlQueueORM.id == item_id)
            .values(**values)
        )

    async def count_by_status(self) -> dict[str, int]:
        """Return count per status — used by notify task to build summary."""
        from sqlalchemy import func
        stmt = (
            select(CrawlQueueORM.status, func.count().label("n"))
            .group_by(CrawlQueueORM.status)
        )
        result = await self.session.execute(stmt)
        return {row.status: row.n for row in result}
