"""Repository for ChordAttempt — extends BaseRepository, adds session and user-scoped queries."""
from __future__ import annotations
from datetime import date, datetime, timezone
from sqlalchemy import select

from src.shared.infrastructure.postgres.base_repo import BaseRepository
from src.user_analytics.domain.models.chord_attempt import ChordAttempt
from src.user_analytics.infrastructure.orm.chord_attempt_orm import ChordAttemptORM


class ChordAttemptRepository(BaseRepository[ChordAttemptORM]):
    """Persists ChordAttempt records from Practice Mode sessions.

    Inherits save, get_by_id, delete_by_id, list_all from BaseRepository.
    """

    model = ChordAttemptORM

    async def save_domain(self, entity: ChordAttempt) -> None:
        """Persist a new ChordAttempt (insert only — attempts are immutable)."""
        await self.save(ChordAttemptORM.from_domain(entity))

    async def get_by_session(self, session_id: str) -> list[ChordAttempt]:
        """Fetch all attempts for a practice session (for session summary)."""
        stmt = select(ChordAttemptORM).where(
            ChordAttemptORM.session_id == session_id
        )
        result = await self.session.execute(stmt)
        return [row.to_domain() for row in result.scalars().all()]

    async def get_by_user_and_chord(
        self, user_id: str, chord: str, limit: int = 100
    ) -> list[ChordAttempt]:
        """Fetch recent attempts for a specific (user, chord) pair.

        Used by dag_analytics_rollup to compute daily accuracy.
        """
        stmt = (
            select(ChordAttemptORM)
            .where(
                ChordAttemptORM.user_id == user_id,
                ChordAttemptORM.target_chord == chord,
            )
            .order_by(ChordAttemptORM.created_at.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return [row.to_domain() for row in result.scalars().all()]

    async def list_by_date(self, target_date: date) -> list[ChordAttempt]:
        """Fetch all attempts for a given date using SQL WHERE — avoids full table scan.

        Used by dag_analytics_rollup instead of list_all() + Python filter.
        """
        day_start = datetime(target_date.year, target_date.month, target_date.day,
                             tzinfo=timezone.utc)
        day_end   = datetime(target_date.year, target_date.month, target_date.day,
                             23, 59, 59, tzinfo=timezone.utc)
        stmt = select(ChordAttemptORM).where(
            ChordAttemptORM.created_at >= day_start,
            ChordAttemptORM.created_at <= day_end,
        )
        result = await self.session.execute(stmt)
        return [row.to_domain() for row in result.scalars().all()]
