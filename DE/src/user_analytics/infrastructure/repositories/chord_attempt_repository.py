"""Repository for chord_attempts table — raw SQL via psycopg2."""
from __future__ import annotations
from datetime import date, datetime, timezone
import logging

from src.shared.infrastructure.postgres.client import get_connection
from src.user_analytics.domain.models.chord_attempt import ChordAttempt

log = logging.getLogger(__name__)


class ChordAttemptRepository:
    """Sync repository — raw SQL, no ORM."""

    def save_domain(self, entity: ChordAttempt) -> None:
        """Insert a new ChordAttempt record."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO chord_attempts
                        (id, user_id, chord_label, is_correct, confidence,
                         response_time, attempted_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    entity.id,
                    entity.user_id,
                    entity.target_chord,
                    entity.is_correct,
                    entity.avg_confidence,
                    0,  # response_time — not tracked yet
                    entity.created_at,
                ))

    def list_by_date(self, target_date: date) -> list[ChordAttempt]:
        """Fetch all attempts for a given date."""
        day_start = datetime(target_date.year, target_date.month, target_date.day,
                             tzinfo=timezone.utc)
        day_end   = datetime(target_date.year, target_date.month, target_date.day,
                             23, 59, 59, tzinfo=timezone.utc)
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, user_id, chord_label, is_correct, confidence,
                           response_time, attempted_at
                    FROM chord_attempts
                    WHERE attempted_at >= %s AND attempted_at <= %s
                """, (day_start, day_end))
                return [self._row_to_domain(row) for row in cur.fetchall()]

    @staticmethod
    def _row_to_domain(row: tuple) -> ChordAttempt:
        return ChordAttempt(
            id=row[0],
            user_id=row[1],
            target_chord=row[2],
            is_correct=row[3],
            root_conf=row[4] or 0.0,
            created_at=row[6],
        )
