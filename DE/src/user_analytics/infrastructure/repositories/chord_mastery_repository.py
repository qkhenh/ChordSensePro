"""Repository for ChordMastery — upsert by composite key (user_id, chord, date), user mastery queries."""
from __future__ import annotations
from datetime import date
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.user_analytics.domain.models.chord_mastery import ChordMastery
from src.user_analytics.infrastructure.orm.chord_mastery_orm import ChordMasteryORM


class ChordMasteryRepository:
    """Persists and queries ChordMastery records.

    Does NOT extend BaseRepository — composite PK (user_id, chord, date)
    requires PostgreSQL upsert (ON CONFLICT DO UPDATE) instead of add/flush.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert(self, entity: ChordMastery) -> None:
        """Insert or update mastery record for (user_id, chord, date).

        Uses PostgreSQL ON CONFLICT DO UPDATE for atomic upsert.
        Called by dag_analytics_rollup after computing rolling accuracy.
        """
        stmt = pg_insert(ChordMasteryORM).values(
            user_id=entity.user_id,
            chord=entity.chord,
            date=entity.date,
            accuracy_today=entity.accuracy_today,
            rolling_accuracy_3d=entity.rolling_accuracy_3d,
            is_mastered=entity.is_mastered,
        ).on_conflict_do_update(
            index_elements=["user_id", "chord", "date"],
            set_={
                "accuracy_today":      entity.accuracy_today,
                "rolling_accuracy_3d": entity.rolling_accuracy_3d,
                "is_mastered":         entity.is_mastered,
            },
        )
        await self._session.execute(stmt)
        await self._session.flush()

    async def get_by_user(self, user_id: str) -> list[ChordMastery]:
        """Fetch all mastery records for a user (used to build learning_plan)."""
        stmt = select(ChordMasteryORM).where(
            ChordMasteryORM.user_id == user_id
        )
        result = await self._session.execute(stmt)
        return [row.to_domain() for row in result.scalars().all()]

    async def get_mastered_chords(self, user_id: str) -> set[str]:
        """Return set of chord labels the user has mastered (accuracy_3d >= 80%)."""
        records = await self.get_by_user(user_id)
        return {r.chord for r in records if r.is_mastered}

    async def get_by_date(self, user_id: str, target_date: date) -> list[ChordMastery]:
        """Fetch mastery snapshot for a specific date (for progress history)."""
        stmt = select(ChordMasteryORM).where(
            ChordMasteryORM.user_id == user_id,
            ChordMasteryORM.date == target_date,
        )
        result = await self._session.execute(stmt)
        return [row.to_domain() for row in result.scalars().all()]
