"""Unit tests for song_analysis domain — no database required."""
import pytest
from datetime import datetime, timezone

from src.song_analysis.domain.models.song_analysis import SongAnalysis, AnalysisStatus


class TestSongAnalysis:
    def test_default_status_is_pending(self):
        sa = SongAnalysis()
        assert sa.status == AnalysisStatus.PENDING

    def test_mark_processing(self):
        sa = SongAnalysis()
        old = sa.updated_at
        sa.mark_processing()
        assert sa.status == AnalysisStatus.PROCESSING
        assert sa.updated_at >= old

    def test_mark_done(self):
        sa = SongAnalysis()
        sa.mark_done(
            key="Am",
            bpm=120.0,
            timeline=[{"chord": "Am", "start_ms": 0}],
            sheet={"sections": []},
            plan=[{"chord": "Am", "status": "mastered"}],
        )
        assert sa.status == AnalysisStatus.DONE
        assert sa.is_done is True
        assert sa.detected_key == "Am"
        assert sa.tempo_bpm == 120.0
        assert len(sa.chord_timeline) == 1

    def test_mark_error(self):
        sa = SongAnalysis()
        sa.mark_error("download failed")
        assert sa.status == AnalysisStatus.ERROR
        assert sa.is_done is False
        assert sa.error_message == "download failed"

    def test_is_done_false_when_pending(self):
        sa = SongAnalysis()
        assert sa.is_done is False


class TestAnalysisStatus:
    def test_status_values(self):
        assert AnalysisStatus.PENDING.value == "pending"
        assert AnalysisStatus.PROCESSING.value == "processing"
        assert AnalysisStatus.DONE.value == "done"
        assert AnalysisStatus.ERROR.value == "error"

    def test_status_from_string(self):
        assert AnalysisStatus("done") == AnalysisStatus.DONE
