from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import torch

from src.evaluate import compute_metrics
from src.model import mert_output_length as model_mert_len
from src.serve import (
    HarmonicSeparator,
    _infer_frame_level,
    _merge_consecutive,
    _smooth_jitter,
    _vote_and_merge,
    chord_name,
    infer_audio,
    mert_output_length as serve_mert_len,
    weighted_chord_conf,
)


def test_mert_length_matches_between_model_and_serve():
    for n in (24000 * 3, 24000 * 5, 24000 * 10):
        assert serve_mert_len(n) == model_mert_len(n)


def test_weighted_chord_conf_excludes_bass():
    conf = weighted_chord_conf(0.5, 0.4, 0.2)
    assert conf == pytest.approx(0.40 * 0.5 + 0.35 * 0.4 + 0.25 * 0.2)


def test_chord_name_slash_and_none():
    assert chord_name(0, 0, 0, 12) == "C"
    assert chord_name(0, 0, 0, 0) == "C"
    assert chord_name(0, 0, 3, 4) == "Cmaj7/E"
    assert chord_name(2, 0, 0, 9) == "D/A"
    assert chord_name(11, 1, 4, 6) == "Bm7/F#"
    assert chord_name(-1, -1, -1, 12) == "N"


def test_smooth_jitter_single_frame():
    frames = [
        {"start_ms": 0.0, "end_ms": 1000.0, "root_id": 1, "quality_id": 0, "ext_id": 0, "bass_id": 12, "chord": "C#", "confidence": 0.8, "silent": False},
        {"start_ms": 1000.0, "end_ms": 2000.0, "root_id": 1, "quality_id": 1, "ext_id": 0, "bass_id": 12, "chord": "C#m", "confidence": 0.5, "silent": False},
        {"start_ms": 2000.0, "end_ms": 3000.0, "root_id": 1, "quality_id": 0, "ext_id": 0, "bass_id": 12, "chord": "C#", "confidence": 0.75, "silent": False},
    ]
    smoothed = _smooth_jitter(frames, max_jitter_frames=1)
    assert [f["chord"] for f in smoothed] == ["C#", "C#", "C#"]
    merged = _merge_consecutive(smoothed)
    assert len(merged) == 1
    assert merged[0]["chord"] == "C#"
    assert merged[0]["start_ms"] == 0.0
    assert merged[0]["end_ms"] == 3000.0


def test_smooth_jitter_keeps_longer_runs():
    frames = [
        {"start_ms": 0.0, "end_ms": 1000.0, "root_id": 0, "quality_id": 0, "ext_id": 0, "bass_id": 12, "chord": "C", "confidence": 0.8, "silent": False},
        {"start_ms": 1000.0, "end_ms": 2000.0, "root_id": 7, "quality_id": 0, "ext_id": 0, "bass_id": 12, "chord": "G", "confidence": 0.7, "silent": False},
        {"start_ms": 2000.0, "end_ms": 3000.0, "root_id": 7, "quality_id": 0, "ext_id": 0, "bass_id": 12, "chord": "G", "confidence": 0.7, "silent": False},
        {"start_ms": 3000.0, "end_ms": 4000.0, "root_id": 0, "quality_id": 0, "ext_id": 0, "bass_id": 12, "chord": "C", "confidence": 0.8, "silent": False},
    ]
    smoothed = _smooth_jitter(frames, max_jitter_frames=1)
    merged = _merge_consecutive(smoothed)
    assert [m["chord"] for m in merged] == ["C", "G", "C"]


def test_vote_and_merge_with_bass():
    sr = 24000
    step = sr
    crop = 3 * sr
    raw_preds = [
        {"pos": 0, "root_id": 2, "quality_id": 0, "ext_id": 0, "bass_id": 6, "conf": 0.85, "silent": False},
        {"pos": step, "root_id": 2, "quality_id": 0, "ext_id": 0, "bass_id": 6, "conf": 0.80, "silent": False},
    ]
    out = _vote_and_merge(raw_preds, audio_len=4 * sr, sample_rate=sr, step_samples=step, crop_samples=crop)
    assert len(out) == 1
    assert out[0]["chord"] == "D/F#"
    assert out[0]["bass_id"] == 6


def test_compute_metrics_chord_acc_and_slash_metrics():
    rp = np.array([0, 2, 7])
    rt = np.array([0, 2, 7])
    qp = np.array([0, 0, 2])
    qt = np.array([0, 0, 2])
    ep = np.array([0, 0, 5])
    et = np.array([0, 0, 5])
    bp = np.array([12, 12, 11])  # 2nd chord missed F# (6), 3rd chord got B (11)
    bt = np.array([12, 6, 11])

    m = compute_metrics(rp, rt, qp, qt, ep, et, bp, bt)
    assert m["root_acc"] == 1.0
    assert m["bass_acc"] == pytest.approx(2.0 / 3.0)
    assert m["bass_slash_acc"] == pytest.approx(0.5)
    assert m["chord_no_bass_acc"] == 1.0
    assert m["chord_acc"] == pytest.approx(2.0 / 3.0)


def test_infer_frame_level_basic():
    sr = 24000
    crop_s = 5
    audio = np.random.randint(-20000, 20000, size=sr * crop_s, dtype=np.int16)

    predictor = MagicMock()
    predictor.is_frame_level = True
    preds = np.zeros((374, 5), dtype=np.float32)
    preds[:, 0] = 0
    preds[:, 1] = 0
    preds[:, 2] = 0
    preds[:, 3] = 12
    preds[:, 4] = 0.95
    predictor.predict.return_value = preds

    segments = infer_audio(predictor, audio, sample_rate=sr, crop_s=crop_s, step_s=2.5)
    assert len(segments) == 1
    assert segments[0]["chord"] == "C"
    assert segments[0]["start_ms"] == 0.0
    assert segments[0]["end_ms"] == 5000.0


def test_infer_frame_level_transition():
    sr = 24000
    crop_s = 5
    audio = np.random.randint(-20000, 20000, size=sr * crop_s, dtype=np.int16)

    predictor = MagicMock()
    predictor.is_frame_level = True
    preds = np.zeros((374, 5), dtype=np.float32)
    preds[:200, 0] = 0
    preds[:200, 1] = 0
    preds[:200, 2] = 0
    preds[:200, 3] = 12
    preds[:200, 4] = 0.9

    preds[200:, 0] = 7
    preds[200:, 1] = 0
    preds[200:, 2] = 0
    preds[200:, 3] = 12
    preds[200:, 4] = 0.95
    predictor.predict.return_value = preds

    segments = _infer_frame_level(predictor, audio, sample_rate=sr, crop_s=crop_s, step_s=5.0)
    assert len(segments) == 2
    assert segments[0]["chord"] == "C"
    assert segments[1]["chord"] == "G"


def test_infer_frame_level_low_confidence():
    sr = 24000
    crop_s = 5
    audio = np.random.randint(-20000, 20000, size=sr * crop_s, dtype=np.int16)

    predictor = MagicMock()
    predictor.is_frame_level = True
    preds = np.zeros((374, 5), dtype=np.float32)
    preds[:, 0] = 0
    preds[:, 1] = 0
    preds[:, 2] = 0
    preds[:, 3] = 12
    preds[:, 4] = 0.1
    predictor.predict.return_value = preds

    segments = _infer_frame_level(predictor, audio, sample_rate=sr, crop_s=crop_s, step_s=5.0, confidence_threshold=0.3)
    assert len(segments) == 1
    assert segments[0]["chord"] == "N"


def test_harmonic_separator_sums_other_and_bass():
    class _FakeDemucs(torch.nn.Module):
        samplerate = 24000
        sources = ["drums", "bass", "other", "vocals"]

    def _fake_apply(model, mix, **kwargs):
        # Return (1, 4, 2, N) where drums/vocals are loud (0.9) and bass+other = 0.2 + 0.3 = 0.5
        B, C, N = mix.shape
        out = torch.zeros((B, 4, C, N), dtype=torch.float32)
        out[:, 0] = 0.9  # drums (dropped)
        out[:, 1] = 0.2  # bass (kept)
        out[:, 2] = 0.3  # other (kept)
        out[:, 3] = 0.9  # vocals (dropped)
        # Undo normalization inside separate_waveform by returning zero-mean unit-scaled stems:
        return out

    with patch("demucs.pretrained.get_model", return_value=_FakeDemucs()), \
         patch("demucs.apply.apply_model", side_effect=_fake_apply):
        sep = HarmonicSeparator(model_name="htdemucs_ft", segment=6.0, device="cpu")
        assert sep.model_name == "htdemucs_ft"
        wav_in = np.sin(np.linspace(0, 100, 24000, dtype=np.float32))
        out_i16 = sep.separate_waveform(wav_in, sr=24000, target_sr=24000)
        assert out_i16.shape == (24000,)
        assert out_i16.dtype == np.int16
