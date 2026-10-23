"""Tests for SeparateHandler._mix_harmonic: piano + guitar + other summed, drums/bass/vocals excluded."""
import pytest

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")

from src.data_processing.application.pipeline_handlers.separate_handler import SeparateHandler


def _write(path, value: float, n: int = 2400, sr: int = 24000):
    sf.write(str(path), np.full((n, 2), value, dtype="float32"), sr)


def test_mix_sums_harmonic_stems_only(tmp_path):
    _write(tmp_path / "piano.wav", 0.1)
    _write(tmp_path / "guitar.wav", 0.2)
    _write(tmp_path / "other.wav", 0.3)
    _write(tmp_path / "drums.wav", 0.9)     # must be ignored
    _write(tmp_path / "vocals.wav", 0.9)    # must be ignored

    res = SeparateHandler()._mix_harmonic(tmp_path)

    assert not res.is_err
    audio, sr = sf.read(str(res.unwrap()), dtype="float32", always_2d=True)
    assert sr == 24000
    assert audio.max() == pytest.approx(0.6, abs=1e-3)


def test_mix_caps_peak(tmp_path):
    _write(tmp_path / "piano.wav", 0.8)
    _write(tmp_path / "guitar.wav", 0.8)

    res = SeparateHandler()._mix_harmonic(tmp_path)

    audio, _ = sf.read(str(res.unwrap()), dtype="float32", always_2d=True)
    assert float(np.abs(audio).max()) <= 0.99 + 1e-3


def test_mix_works_with_partial_stem_set(tmp_path):
    _write(tmp_path / "other.wav", 0.25)

    res = SeparateHandler()._mix_harmonic(tmp_path)

    assert not res.is_err
    assert res.unwrap().name == "harmonic.wav"


def test_mix_errors_when_no_harmonic_stem(tmp_path):
    _write(tmp_path / "drums.wav", 0.5)

    res = SeparateHandler()._mix_harmonic(tmp_path)

    assert res.is_err
    assert "harmonic stems" in res.error
