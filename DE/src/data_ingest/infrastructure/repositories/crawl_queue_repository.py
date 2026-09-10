"""Repository for crawl_queue table — raw SQL via psycopg2."""
from __future__ import annotations
import uuid
import logging
from datetime import datetime, timezone

from src.shared.infrastructure.postgres.client import get_connection
from src.data_ingest.domain.models.crawl_queue_item import CrawlQueueItem, CrawlStatus

log = logging.getLogger(__name__)


class CrawlQueueRepository:
    """Sync repository — raw SQL, no ORM."""

    def get_pending(self, limit: int = 10) -> list[CrawlQueueItem]:
        """Fetch N pending items ordered by priority DESC, created_at ASC."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, source_url, annotation_url, source_type,
                           status, priority, error_message, created_at, processed_at
                    FROM crawl_queue
                    WHERE status = %s
                    ORDER BY priority DESC, created_at ASC
                    LIMIT %s
                """, (CrawlStatus.PENDING.value, limit))
                return [self._row_to_domain(row) for row in cur.fetchall()]

    def add_url(
        self,
        source_url: str,
        annotation_url: str = "",
        source_type: str = "local",
        priority: int = 0,
    ) -> CrawlQueueItem | None:
        """Insert a URL. Returns None if duplicate (ON CONFLICT DO NOTHING)."""
        item_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO crawl_queue
                        (id, source_url, annotation_url, source_type, status, priority, created_at)
                    VALUES (%s, %s, %s, %s, 'pending', %s, %s)
                    ON CONFLICT (source_url) DO NOTHING
                """, (item_id, source_url, annotation_url, source_type, priority, now))
                if cur.rowcount == 0:
                    return None
        return CrawlQueueItem(
            id=item_id, source_url=source_url, annotation_url=annotation_url,
            source_type=source_type, priority=priority, created_at=now,
        )

    def update_status(
        self,
        item_id: str,
        status: CrawlStatus,
        error_message: str | None = None,
    ) -> None:
        """Update status (and optional error) for one queue item."""
        now = datetime.now(timezone.utc)
        with get_connection() as conn:
            with conn.cursor() as cur:
                if error_message is not None:
                    cur.execute("""
                        UPDATE crawl_queue
                        SET status = %s, error_message = %s, processed_at = %s
                        WHERE id = %s
                    """, (status.value, error_message, now, item_id))
                else:
                    cur.execute("""
                        UPDATE crawl_queue
                        SET status = %s, processed_at = %s
                        WHERE id = %s
                    """, (status.value, now, item_id))

    def count_by_status(self) -> dict[str, int]:
        """Return count per status."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT status, COUNT(*) FROM crawl_queue GROUP BY status
                """)
                return {row[0]: row[1] for row in cur.fetchall()}

    @staticmethod
    def _row_to_domain(row: tuple) -> CrawlQueueItem:
        """Map DB row → domain entity."""
        return CrawlQueueItem(
            id=row[0],
            source_url=row[1],
            annotation_url=row[2] or "",
            source_type=row[3] or "local",
            status=CrawlStatus(row[4]),
            priority=row[5],
            error_message=row[6],
            created_at=row[7],
            processed_at=row[8],
        )
