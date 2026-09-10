"""Repository for song_analyses table — raw SQL via psycopg2."""
from __future__ import annotations
import json
import logging

from src.shared.infrastructure.postgres.client import get_connection
from src.shared.domain.processing_result import ProcessingResult
from src.song_analysis.domain.models.song_analysis import SongAnalysis, AnalysisStatus

log = logging.getLogger(__name__)


class SongAnalysisRepository:
    """Sync repository — raw SQL, no ORM."""

    def save_domain(self, entity: SongAnalysis) -> None:
        """Upsert SongAnalysis — INSERT ... ON CONFLICT UPDATE."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO song_analyses
                        (id, user_id, source_url, song_title, status,
                         detected_key, tempo_bpm, chord_timeline, chord_sheet,
                         learning_plan, error_message, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        status = EXCLUDED.status,
                        detected_key = EXCLUDED.detected_key,
                        tempo_bpm = EXCLUDED.tempo_bpm,
                        chord_timeline = EXCLUDED.chord_timeline,
                        chord_sheet = EXCLUDED.chord_sheet,
                        learning_plan = EXCLUDED.learning_plan,
                        error_message = EXCLUDED.error_message,
                        updated_at = EXCLUDED.updated_at
                """, (
                    entity.id,
                    entity.user_id,
                    entity.source_url,
                    entity.song_title,
                    entity.status.value,
                    entity.detected_key,
                    entity.tempo_bpm,
                    json.dumps(entity.chord_timeline or []),
                    json.dumps(entity.chord_sheet or {}),
                    json.dumps(entity.learning_plan or []),
                    entity.error_message,
                    entity.created_at,
                    entity.updated_at,
                ))

    def get_domain_by_id(self, analysis_id: str) -> ProcessingResult[SongAnalysis]:
        """Fetch by UUID. Returns Err if not found."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, user_id, source_url, song_title, status,
                           detected_key, tempo_bpm, chord_timeline, chord_sheet,
                           learning_plan, error_message, created_at, updated_at
                    FROM song_analyses WHERE id = %s
                """, (analysis_id,))
                row = cur.fetchone()
                if row is None:
                    return ProcessingResult.err(f"SongAnalysis not found: {analysis_id}")
                return ProcessingResult.ok(self._row_to_domain(row))

    def get_by_url(self, source_url: str) -> ProcessingResult[SongAnalysis]:
        """Fetch a completed analysis by source URL (cache check)."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, user_id, source_url, song_title, status,
                           detected_key, tempo_bpm, chord_timeline, chord_sheet,
                           learning_plan, error_message, created_at, updated_at
                    FROM song_analyses
                    WHERE source_url = %s AND status = 'done'
                """, (source_url,))
                row = cur.fetchone()
                if row is None:
                    return ProcessingResult.err(f"No cached analysis for: {source_url}")
                return ProcessingResult.ok(self._row_to_domain(row))

    @staticmethod
    def _row_to_domain(row: tuple) -> SongAnalysis:
        """Map DB row → domain entity."""
        return SongAnalysis(
            id=row[0],
            user_id=row[1],
            source_url=row[2],
            song_title=row[3],
            status=AnalysisStatus(row[4]),
            detected_key=row[5],
            tempo_bpm=row[6],
            chord_timeline=row[7] or [],
            chord_sheet=row[8] or {},
            learning_plan=row[9] or [],
            error_message=row[10],
            created_at=row[11],
            updated_at=row[12],
        )
