"""SQLAlchemy ORM model for crawl_queue table."""
from __future__ import annotations
from datetime import datetime
from sqlalchemy import String, Integer, Text, DateTime
from sqlalchemy.orm import Mapped, mapped_column

from src.shared.infrastructure.postgres.orm_base import Base
from src.data_ingest.domain.models.crawl_queue_item import CrawlQueueItem, CrawlStatus


class CrawlQueueORM(Base):
    """SQLAlchemy ORM model — maps to PostgreSQL crawl_queue table."""
    __tablename__ = "crawl_queue"

    id:            Mapped[str]           = mapped_column(String(36), primary_key=True)
    source_url:    Mapped[str]           = mapped_column(Text, unique=True, index=True)
    status:        Mapped[str]           = mapped_column(String(20), default="pending", index=True)
    priority:      Mapped[int]           = mapped_column(Integer, default=0)
    error_message: Mapped[str | None]    = mapped_column(Text, nullable=True)
    created_at:    Mapped[datetime]      = mapped_column(DateTime(timezone=True))
    processed_at:  Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    def to_domain(self) -> CrawlQueueItem:
        """ORM row → domain entity."""
        return CrawlQueueItem(
            id=self.id,
            source_url=self.source_url,
            status=CrawlStatus(self.status),
            priority=self.priority,
            error_message=self.error_message,
            created_at=self.created_at,
            processed_at=self.processed_at,
        )

    @staticmethod
    def from_domain(entity: CrawlQueueItem) -> CrawlQueueORM:
        """Domain entity → ORM row."""
        return CrawlQueueORM(
            id=entity.id,
            source_url=entity.source_url,
            status=entity.status.value,
            priority=entity.priority,
            error_message=entity.error_message,
            created_at=entity.created_at,
            processed_at=entity.processed_at,
        )
