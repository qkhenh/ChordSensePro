"""Unit tests for AudioSegment entity — no librosa or database required."""
import pytest

from src.data_processing.domain.models.audio_segment import AudioSegment


class TestAudioSegment:
    def test_duration_ms(self):
        seg = AudioSegment(start_ms=1000.0, end_ms=3000.0)
        assert seg.duration_ms == 2000.0

    def test_has_features_false_by_default(self):
        seg = AudioSegment()
        assert seg.has_features is False

    def test_has_features_true_when_all_filled(self):
        seg = AudioSegment(
            chroma_cqt=[0.0] * 12,
            chroma_cens=[0.0] * 12,
            hpss_harmonic=[0.0] * 64,
            mel_high=[0.0] * 64,
        )
        assert seg.has_features is True

    def test_has_features_false_when_partial(self):
        """Missing one channel → has_features should be False."""
        seg = AudioSegment(
            chroma_cqt=[0.0] * 12,
            chroma_cens=[0.0] * 12,
            # hpss_harmonic missing
            mel_high=[0.0] * 64,
        )
        assert seg.has_features is False

    def test_default_sample_rate(self):
        seg = AudioSegment()
        assert seg.sample_rate == 44100
