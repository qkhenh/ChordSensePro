from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.augment import Augmenter, pitch_shift_bass, pitch_shift_root
from src.config import AugmentCfg


def test_pitch_shift_root_all():
    for root in range(12):
        for shift in range(-5, 7):
            result = pitch_shift_root(root, shift)
            assert 0 <= result < 12
            assert result == (root + shift) % 12


def test_pitch_shift_bass():
    for bass in range(12):
        for shift in range(-5, 7):
            result = pitch_shift_bass(bass, shift)
            assert 0 <= result < 12
            assert result == (bass + shift) % 12
    assert pitch_shift_bass(12, 4) == 12
    assert pitch_shift_bass(-100, 4) == -100


def test_pitch_shift_root_ignore():
    assert pitch_shift_root(-100, 3) == -100
    assert pitch_shift_root(-1, 5) == -1


def test_augmenter_disabled():
    cfg = AugmentCfg(enabled=False)
    aug = Augmenter(cfg)
    audio = np.random.randn(120000).astype(np.float32)
    out, shift = aug(audio, 24000)
    assert shift == 0
    np.testing.assert_array_equal(out, audio)


def test_augmenter_output_shape():
    cfg = AugmentCfg(enabled=True, pitch_shift_range=[0, 0], noise_prob=0.0, reverb_prob=0.0)
    aug = Augmenter(cfg)
    audio = np.random.randn(120000).astype(np.float32) * 0.1
    out, shift = aug(audio, 24000)
    assert out.shape == audio.shape
    assert shift == 0


def test_augmenter_gain_changes_amplitude():
    cfg = AugmentCfg(enabled=True, pitch_shift_range=[0, 0], noise_prob=0.0, gain_db=12.0)
    aug = Augmenter(cfg)
    audio = np.ones(24000, dtype=np.float32) * 0.5
    np.random.seed(99)
    out, _ = aug(audio, 24000)
    assert out.shape == audio.shape


def test_augmenter_noise():
    cfg = AugmentCfg(enabled=True, pitch_shift_range=[0, 0], noise_prob=1.0, noise_snr_range=[5, 5])
    aug = Augmenter(cfg)
    audio = np.ones(24000, dtype=np.float32) * 0.5
    out, _ = aug(audio, 24000)
    assert not np.allclose(out, audio * (10.0 ** (0.0 / 20.0)), atol=1e-4)


def test_augmenter_time_mask():
    cfg = AugmentCfg(
        enabled=True,
        pitch_shift_range=[0, 0],
        gain_db=0.0,
        noise_prob=0.0,
        time_mask_prob=1.0,
        time_mask_max_ms=120,
        n_time_masks=2,
    )
    aug = Augmenter(cfg)
    audio = np.ones(24000 * 5, dtype=np.float32)
    out, shift = aug(audio, 24000)
    assert shift == 0
    assert (out == 0.0).sum() >= int(0.06 * 24000)
