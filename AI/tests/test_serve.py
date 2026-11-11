from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.evaluate import compute_metrics
from src.serve import _merge_consecutive, _smooth_jitter, _vote_and_merge, chord_name


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


def test_compute_metrics_chord_acc_includes_bass():
    rp = np.array([0, 2])
    rt = np.array([0, 2])
    qp = np.array([0, 0])
    qt = np.array([0, 0])
    ep = np.array([0, 0])
    et = np.array([0, 0])
    bp = np.array([12, 12])  # second chord predicted no-slash (12), true is F# (6)
    bt = np.array([12, 6])

    m = compute_metrics(rp, rt, qp, qt, ep, et, bp, bt)
    assert m["root_acc"] == 1.0
    assert m["bass_acc"] == 0.5
    assert m["chord_acc"] == 0.5


def test_infer_frame_level_basic():
    from unittest.mock import MagicMock
    from src.serve import infer_audio

    sr = 24000
    crop_s = 5
    audio = np.random.randint(-20000, 20000, size=sr * crop_s, dtype=np.int16)

    predictor = MagicMock()
    predictor.is_frame_level = True
    preds = np.zeros((375, 5), dtype=np.float32)
    preds[:, 0] = 0
    preds[:, 1] = 0
    preds[:, 2] = 0
    preds[:, 3] = 12
    preds[:, 4] = 0.95
    predictor.predict.return_value = preds

    segments = infer_audio(predictor, audio, sample_rate=sr, crop_s=crop_s, step_s=1.0)
    assert len(segments) == 1
    assert segments[0]["chord"] == "C"
    assert segments[0]["start_ms"] == 0.0
    assert segments[0]["end_ms"] == 5000.0


def test_infer_frame_level_transition():
    from unittest.mock import MagicMock
    from src.serve import _infer_frame_level

    sr = 24000
    crop_s = 5
    audio = np.random.randint(-20000, 20000, size=sr * crop_s, dtype=np.int16)

    predictor = MagicMock()
    predictor.is_frame_level = True
    preds = np.zeros((375, 5), dtype=np.float32)
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
    from unittest.mock import MagicMock
    from src.serve import _infer_frame_level

    sr = 24000
    crop_s = 5
    audio = np.random.randint(-20000, 20000, size=sr * crop_s, dtype=np.int16)

    predictor = MagicMock()
    predictor.is_frame_level = True
    preds = np.zeros((375, 5), dtype=np.float32)
    preds[:, 0] = 0
    preds[:, 1] = 0
    preds[:, 2] = 0
    preds[:, 3] = 12
    preds[:, 4] = 0.1
    predictor.predict.return_value = preds

    segments = _infer_frame_level(predictor, audio, sample_rate=sr, crop_s=crop_s, step_s=5.0, confidence_threshold=0.3)
    assert len(segments) == 1
    assert segments[0]["chord"] == "N"


