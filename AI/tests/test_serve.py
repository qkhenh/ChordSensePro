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
    _prune_short_segments,
    _smooth_jitter,
    _vote_and_merge,
    apply_key_spelling,
    canonical_bass_id,
    canonicalize_chord_tuple,
    chord_name,
    decode_musical_sequence,
    detect_prefer_flats,
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


def test_merge_consecutive_folds_root_bass_into_none():
    assert canonical_bass_id(9, 9) == 12
    assert canonical_bass_id(9, 0) == 0
    segs = [
        {"start_ms": 0.0, "end_ms": 500.0, "root_id": 9, "quality_id": 1, "ext_id": 0, "bass_id": 12, "chord": "Am", "confidence": 0.70, "silent": False},
        {"start_ms": 500.0, "end_ms": 1000.0, "root_id": 9, "quality_id": 1, "ext_id": 0, "bass_id": 9, "chord": "Am", "confidence": 0.80, "silent": False},
        {"start_ms": 1000.0, "end_ms": 1500.0, "root_id": 9, "quality_id": 1, "ext_id": 0, "bass_id": 12, "chord": "Am", "confidence": 0.75, "silent": False},
    ]
    merged = _merge_consecutive(segs)
    assert len(merged) == 1
    assert merged[0]["chord"] == "Am"
    assert merged[0]["bass_id"] == 12
    assert merged[0]["start_ms"] == 0.0
    assert merged[0]["end_ms"] == 1500.0


def test_symmetric_dim7_canonicalization():
    # Adim7/C (9, 3, 5, 0) and F#dim7/C (6, 3, 5, 0) both equal Cdim7 (0, 3, 5, 12)
    assert canonicalize_chord_tuple(9, 3, 5, 0) == (0, 3, 5, 12)
    assert canonicalize_chord_tuple(6, 3, 5, 0) == (0, 3, 5, 12)
    assert canonicalize_chord_tuple(3, 3, 5, 0) == (0, 3, 5, 12)


def test_prune_short_segments_staircase():
    segs = [
        {"start_ms": 0.0, "end_ms": 1200.0, "root_id": 2, "quality_id": 1, "ext_id": 4, "bass_id": 12, "chord": "Dm7", "confidence": 0.85, "silent": False},
        {"start_ms": 1200.0, "end_ms": 1240.0, "root_id": 9, "quality_id": 3, "ext_id": 0, "bass_id": 12, "chord": "Adim", "confidence": 0.52, "silent": False},
        {"start_ms": 1240.0, "end_ms": 1290.0, "root_id": 5, "quality_id": 3, "ext_id": 5, "bass_id": 0, "chord": "Fdim7/C", "confidence": 0.54, "silent": False},
        {"start_ms": 1290.0, "end_ms": 2400.0, "root_id": 5, "quality_id": 2, "ext_id": 5, "bass_id": 12, "chord": "F7", "confidence": 0.92, "silent": False},
    ]
    pruned = _prune_short_segments(segs, min_duration_ms=300.0)
    assert len(pruned) == 2
    assert [p["chord"] for p in pruned] == ["Dm7", "F7"]
    assert all((p["end_ms"] - p["start_ms"]) >= 300.0 for p in pruned)


def test_decode_musical_sequence_suppresses_arpeggio_flicker():
    n = 150  # ~2.0 seconds at 75 Hz
    rp = np.full((n, 12), 0.01, dtype=np.float32)
    qp = np.full((n, 6), 0.02, dtype=np.float32)
    ep = np.full((n, 15), 0.01, dtype=np.float32)
    bp = np.full((n, 13), 0.01, dtype=np.float32)

    # Sustained Dm7 (root=2, qual=1, ext=4, bass=12)
    rp[:, 2] = 0.89
    qp[:, 1] = 0.90
    ep[:, 4] = 0.65
    ep[:, 8] = 0.21  # m9 secondary
    bp[:, 12] = 0.88

    # Inject a 4-frame (~53ms) single-note arpeggio blip on root=9 (A) + m9 flicker
    rp[60:64, 2] = 0.25
    rp[60:64, 9] = 0.64
    ep[30:42, 4] = 0.30
    ep[30:42, 8] = 0.56

    decoded = decode_musical_sequence(rp, qp, ep, bp, frame_ms=13.333)
    assert np.all(decoded[:, 0] == 2)   # Root stays D throughout
    assert np.all(decoded[:, 1] == 1)   # Quality stays minor throughout
    assert np.all(decoded[:, 2] == 4)   # Extension consolidated to m7 without splitting
    assert np.all(decoded[:, 3] == 12)  # Bass stays none


def test_key_aware_flat_vs_sharp_spelling():
    dm_prog = [
        {"start_ms": 0.0, "end_ms": 2000.0, "root_id": 2, "quality_id": 1, "ext_id": 4, "bass_id": 12, "chord": "Dm7", "confidence": 0.85, "silent": False},
        {"start_ms": 2000.0, "end_ms": 4000.0, "root_id": 10, "quality_id": 0, "ext_id": 3, "bass_id": 12, "chord": "A#maj7", "confidence": 0.87, "silent": False},
        {"start_ms": 4000.0, "end_ms": 6000.0, "root_id": 7, "quality_id": 1, "ext_id": 4, "bass_id": 12, "chord": "Gm7", "confidence": 0.90, "silent": False},
        {"start_ms": 6000.0, "end_ms": 8000.0, "root_id": 9, "quality_id": 2, "ext_id": 5, "bass_id": 12, "chord": "A7", "confidence": 0.80, "silent": False},
    ]
    assert detect_prefer_flats(dm_prog) is True
    spelled = apply_key_spelling(dm_prog)
    assert spelled[1]["chord"] == "Bbmaj7"

    e_maj_prog = [
        {"start_ms": 0.0, "end_ms": 2000.0, "root_id": 4, "quality_id": 0, "ext_id": 0, "bass_id": 12, "chord": "E", "confidence": 0.9, "silent": False},
        {"start_ms": 2000.0, "end_ms": 4000.0, "root_id": 11, "quality_id": 0, "ext_id": 0, "bass_id": 12, "chord": "B", "confidence": 0.9, "silent": False},
        {"start_ms": 4000.0, "end_ms": 6000.0, "root_id": 1, "quality_id": 1, "ext_id": 0, "bass_id": 12, "chord": "C#m", "confidence": 0.9, "silent": False},
        {"start_ms": 6000.0, "end_ms": 8000.0, "root_id": 9, "quality_id": 0, "ext_id": 0, "bass_id": 12, "chord": "A", "confidence": 0.9, "silent": False},
    ]
    assert detect_prefer_flats(e_maj_prog) is False
    spelled_e = apply_key_spelling(e_maj_prog)
    assert spelled_e[2]["chord"] == "C#m"

