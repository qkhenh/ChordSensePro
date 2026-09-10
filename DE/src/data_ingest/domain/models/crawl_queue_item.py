"""CrawlQueueItem — domain model representing one URL in the crawl queue."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from src.shared.domain.base_model import BaseEntity


class CrawlStatus(str, Enum):
    PENDING    = "pending"     # waiting to be picked up by DAG
    RUNNING    = "running"     # currently being processed
    DONE       = "done"        # ETL completed successfully
    FAILED     = "failed"      # ETL failed, see error_message
    SKIPPED    = "skipped"     # already processed (dedup)


@dataclass
class CrawlQueueItem(BaseEntity):
    """One URL entry in the crawl queue.

    Inserted manually via scripts/add_urls.py or via API.
    Picked up by dag_crawl_batch every N hours.
    """
    source_url:     str = ""
    annotation_url: str = ""
    source_type:    str = "local"   # jaah | kaggle | choco | local
    status:         CrawlStatus = CrawlStatus.PENDING
    priority:       int         = 0            # higher = picked first
    error_message:  str | None  = None
    processed_at:   datetime | None = None

    def mark_running(self) -> None:
        self.status = CrawlStatus.RUNNING

    def mark_done(self) -> None:
        self.status       = CrawlStatus.DONE
        self.processed_at = datetime.now(timezone.utc)

    def mark_failed(self, error: str) -> None:
        self.status        = CrawlStatus.FAILED
        self.error_message = error
        self.processed_at  = datetime.now(timezone.utc)

    def mark_skipped(self) -> None:
        self.status       = CrawlStatus.SKIPPED
        self.processed_at = datetime.now(timezone.utc)
