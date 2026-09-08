"""SQLAlchemy ORM for user_chord_mastery table — composite PK (user_id, chord, date)."""
from __future__ import annotations
from datetime import date

from sqlalchemy import String, Float, Boolean, Date, PrimaryKeyConstraint
from sqlalchemy.orm import Mapped, mapped_column
from src.shared.infrastructure.postgres.orm_base import Base
from src.user_analytics.domain.models.chord_mastery import ChordMastery


class ChordMasteryORM(Base):
    """ORM model for user_chord_mastery table.

    Uses composite primary key (user_id, chord, date) — no UUID id.
    """

    __tablename__ = "user_chord_mastery"
    __table_args__ = (
        PrimaryKeyConstraint("user_id", "chord", "date"),
    )

    user_id:             Mapped[str]   = mapped_column(String(36))
    chord:               Mapped[str]   = mapped_column(String(20))
    date:                Mapped[date]  = mapped_column(Date)
    accuracy_today:      Mapped[float] = mapped_column(Float, default=0.0)
    rolling_accuracy_3d: Mapped[float] = mapped_column(Float, default=0.0)
    is_mastered:         Mapped[bool]  = mapped_column(Boolean, default=False)

    def to_domain(self) -> ChordMastery:
        return ChordMastery(
            user_id=self.user_id,
            chord=self.chord,
            date=self.date,
            accuracy_today=self.accuracy_today,
            rolling_accuracy_3d=self.rolling_accuracy_3d,
            is_mastered=self.is_mastered,
        )

    @staticmethod
    def from_domain(entity: ChordMastery) -> ChordMasteryORM:
        return ChordMasteryORM(
            user_id=entity.user_id,
            chord=entity.chord,
            date=entity.date,
            accuracy_today=entity.accuracy_today,
            rolling_accuracy_3d=entity.rolling_accuracy_3d,
            is_mastered=entity.is_mastered,
        )
