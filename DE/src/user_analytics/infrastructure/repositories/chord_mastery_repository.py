"""Repository for user_chord_mastery table — raw SQL via psycopg2."""
from __future__ import annotations
from datetime import date
import logging

from src.shared.infrastructure.postgres.client import get_connection
from src.user_analytics.domain.models.chord_mastery import ChordMastery

log = logging.getLogger(__name__)


class ChordMasteryRepository:
    """Sync repository — raw SQL, no ORM."""

    def upsert(self, entity: ChordMastery) -> None:
        """Insert or update mastery record for (user_id, chord_label)."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO user_chord_mastery
                        (user_id, chord_label, mastery_level, total_attempts,
                         correct_count, last_practiced, is_mastered)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_id, chord_label) DO UPDATE SET
                        mastery_level  = EXCLUDED.mastery_level,
                        total_attempts = EXCLUDED.total_attempts,
                        correct_count  = EXCLUDED.correct_count,
                        last_practiced = EXCLUDED.last_practiced,
                        is_mastered    = EXCLUDED.is_mastered
                """, (
                    entity.user_id,
                    entity.chord,
                    entity.rolling_accuracy_3d,
                    0,  # total_attempts — computed at rollup
                    0,  # correct_count — computed at rollup
                    entity.date,
                    entity.is_mastered,
                ))

    def get_by_user(self, user_id: str) -> list[ChordMastery]:
        """Fetch all mastery records for a user."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT user_id, chord_label, mastery_level, total_attempts,
                           correct_count, last_practiced, is_mastered
                    FROM user_chord_mastery
                    WHERE user_id = %s
                """, (user_id,))
                return [self._row_to_domain(row) for row in cur.fetchall()]

    def get_mastered_chords(self, user_id: str) -> set[str]:
        """Return set of chord labels the user has mastered."""
        records = self.get_by_user(user_id)
        return {r.chord for r in records if r.is_mastered}

    @staticmethod
    def _row_to_domain(row: tuple) -> ChordMastery:
        return ChordMastery(
            user_id=row[0],
            chord=row[1],
            date=row[5] or date.today(),
            rolling_accuracy_3d=row[2] or 0.0,
            is_mastered=row[6],
        )
