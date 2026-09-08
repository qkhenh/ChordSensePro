"""Unit tests for user_analytics domain — no database required."""
import pytest
from datetime import date

from src.user_analytics.domain.models.chord_attempt import ChordAttempt
from src.user_analytics.domain.models.chord_mastery import ChordMastery, MASTERY_THRESHOLD


class TestChordAttempt:
    def test_avg_confidence(self):
        attempt = ChordAttempt(
            root_conf=0.9,
            quality_conf=0.8,
            extension_conf=0.7,
        )
        assert abs(attempt.avg_confidence - 0.8) < 1e-9

    def test_is_correct_default_false(self):
        attempt = ChordAttempt()
        assert attempt.is_correct is False

    def test_fields_populated(self):
        attempt = ChordAttempt(
            user_id="u1",
            session_id="s1",
            target_chord="Am7",
            detected_chord="Am",
            is_correct=False,
        )
        assert attempt.target_chord == "Am7"
        assert attempt.detected_chord == "Am"


class TestChordMastery:
    def test_mastery_threshold_is_80_percent(self):
        assert MASTERY_THRESHOLD == 0.80

    def test_recompute_mastery_above_threshold(self):
        m = ChordMastery(
            user_id="u1", chord="Am", date=date.today(),
            rolling_accuracy_3d=0.85,
        )
        m.recompute_mastery()
        assert m.is_mastered is True

    def test_recompute_mastery_below_threshold(self):
        m = ChordMastery(
            user_id="u1", chord="Am", date=date.today(),
            rolling_accuracy_3d=0.75,
        )
        m.recompute_mastery()
        assert m.is_mastered is False

    def test_recompute_mastery_exact_threshold(self):
        m = ChordMastery(
            user_id="u1", chord="Am", date=date.today(),
            rolling_accuracy_3d=0.80,
        )
        m.recompute_mastery()
        assert m.is_mastered is True

    def test_needs_practice(self):
        m = ChordMastery(
            user_id="u1", chord="Dm7", date=date.today(),
            rolling_accuracy_3d=0.50,
            is_mastered=False,
        )
        assert m.needs_practice is True

    def test_needs_practice_false_when_mastered(self):
        m = ChordMastery(
            user_id="u1", chord="C", date=date.today(),
            is_mastered=True,
        )
        assert m.needs_practice is False
