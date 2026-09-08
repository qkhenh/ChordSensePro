"""SQLAlchemy ORM model for song_analyses table with to_domain / from_domain mapper methods."""
from __future__ import annotations
from datetime import datetime
from sqlalchemy import String, Float, Text, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from src.shared.infrastructure.postgres.orm_base import Base
from src.song_analysis.domain.models.song_analysis import SongAnalysis, AnalysisStatus

class SongAnalysisORM(Base):
    """SQLAlchemy ORM model — maps to PostgreSQL song_analyses table."""
    __tablename__ = "song_analyses"
    id:             Mapped[str]           = mapped_column(String(36), primary_key=True)
    user_id:        Mapped[str]           = mapped_column(String(36), index=True)
    source_url:     Mapped[str]           = mapped_column(Text, index=True)
    song_title:     Mapped[str]           = mapped_column(String(200), default="")
    status:         Mapped[str]           = mapped_column(String(20), default="pending")
    detected_key:   Mapped[str | None]    = mapped_column(String(10), nullable=True)
    tempo_bpm:      Mapped[float | None]  = mapped_column(Float, nullable=True)
    chord_timeline: Mapped[list]          = mapped_column(JSONB, default=list)
    chord_sheet:    Mapped[dict]          = mapped_column(JSONB, default=dict)
    learning_plan:  Mapped[list]          = mapped_column(JSONB, default=list)
    error_message:  Mapped[str | None]    = mapped_column(Text, nullable=True)
    created_at:     Mapped[datetime]      = mapped_column(DateTime(timezone=True))
    updated_at:     Mapped[datetime]      = mapped_column(DateTime(timezone=True))
    
    def to_domain(self) -> SongAnalysis:
        """ORM row → domain entity."""
        return SongAnalysis(
            id = self.id,
            user_id = self.user_id,
            source_url = self.source_url,
            song_title = self.song_title,
            status = AnalysisStatus(self.status),
            detected_key = self.detected_key,
            tempo_bpm = self.tempo_bpm,
            chord_timeline = self.chord_timeline or [],
            chord_sheet = self.chord_sheet or {},
            learning_plan = self.learning_plan or [],
            error_message =self.error_message,
            created_at = self.created_at,
            updated_at = self.updated_at,
        )
        
    @staticmethod
    def from_domain(entity: SongAnalysis) -> SongAnalysisORM:
        """Domain entity → ORM row."""
        return SongAnalysisORM(
            id = entity.id,
            user_id = entity.user_id,
            source_url = entity.source_url,
            song_title = entity.song_title,
            status = entity.status.value,
            detected_key = entity.detected_key,
            tempo_bpm = entity.tempo_bpm,
            chord_timeline = entity.chord_timeline,
            chord_sheet = entity.chord_sheet,
            learning_plan = entity.learning_plan,
            error_message = entity.error_message,
            created_at = entity.created_at,
            updated_at = entity.updated_at,
        )