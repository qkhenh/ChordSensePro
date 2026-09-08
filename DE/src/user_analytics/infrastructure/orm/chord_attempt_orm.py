"""SQLAlchemy ORM for chord_attempts table with to_domain / from_domain mapper methods."""
from __future__ import annotations
from datetime import datetime

from sqlalchemy import String, Float, Boolean, DateTime, Index
from sqlalchemy.orm import Mapped, mapped_column
from src.shared.infrastructure.postgres.orm_base import Base
from src.user_analytics.domain.models.chord_attempt import ChordAttempt


class ChordAttemptORM(Base):
    """ORM model for chord_attempts table."""

    __tablename__ = "chord_attempts"
    __table_args__ = (
        # Composite index for dag_analytics_rollup — avoids full scan on get_by_user_and_chord()
        Index("ix_chord_attempts_user_chord_created", "user_id", "target_chord", "created_at"),
    )

    id:             Mapped[str]   = mapped_column(String(36), primary_key=True)
    user_id:        Mapped[str]   = mapped_column(String(36), index=True)
    session_id:     Mapped[str]   = mapped_column(String(36), index=True)
    target_chord:   Mapped[str]   = mapped_column(String(20))
    detected_chord: Mapped[str]   = mapped_column(String(20))
    root_conf:      Mapped[float] = mapped_column(Float, default=0.0)
    quality_conf:   Mapped[float] = mapped_column(Float, default=0.0)
    extension_conf: Mapped[float] = mapped_column(Float, default=0.0)
    is_correct:     Mapped[bool]  = mapped_column(Boolean, default=False)
    created_at:     Mapped[datetime] = mapped_column(DateTime(timezone=True))

    def to_domain(self) -> ChordAttempt:
        return ChordAttempt(
            id=self.id,
            created_at=self.created_at,
            user_id=self.user_id,
            session_id=self.session_id,
            target_chord=self.target_chord,
            detected_chord=self.detected_chord,
            root_conf=self.root_conf,
            quality_conf=self.quality_conf,
            extension_conf=self.extension_conf,
            is_correct=self.is_correct,
        )

    @staticmethod
    def from_domain(entity: ChordAttempt) -> ChordAttemptORM:
        return ChordAttemptORM(
            id=entity.id,
            created_at=entity.created_at,
            user_id=entity.user_id,
            session_id=entity.session_id,
            target_chord=entity.target_chord,
            detected_chord=entity.detected_chord,
            root_conf=entity.root_conf,
            quality_conf=entity.quality_conf,
            extension_conf=entity.extension_conf,
            is_correct=entity.is_correct,
        )
