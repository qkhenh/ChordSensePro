from __future__ import annotations

import numpy as np
import torch
import torchaudio.functional as AF

from src.config import AugmentCfg


class Augmenter:

    def __init__(self, cfg: AugmentCfg):
        self.cfg = cfg

    def __call__(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        pitch_shift_n = 0

        if self.cfg.enabled:
            lo, hi = self.cfg.pitch_shift_range
            if lo != 0 or hi != 0:
                pitch_shift_n = int(np.random.randint(lo, hi + 1))
                if pitch_shift_n != 0:
                    audio = self._pitch_shift(audio, sr, pitch_shift_n)

            audio = self._gain(audio)

            if np.random.random() < self.cfg.noise_prob:
                audio = self._add_noise(audio)

            if getattr(self.cfg, "time_mask_prob", 0.0) > 0.0 and np.random.random() < self.cfg.time_mask_prob:
                audio = self._time_mask(audio, sr)

        return audio, pitch_shift_n

    def _pitch_shift(self, audio: np.ndarray, sr: int, n_steps: int) -> np.ndarray:
        wav_t = torch.from_numpy(audio).unsqueeze(0)
        shifted = AF.pitch_shift(wav_t, sr, n_steps)
        return shifted.squeeze(0).numpy()

    def _gain(self, audio: np.ndarray) -> np.ndarray:
        gain_db = np.random.uniform(-self.cfg.gain_db, self.cfg.gain_db)
        return (audio * (10.0 ** (gain_db / 20.0))).astype(np.float32)

    def _add_noise(self, audio: np.ndarray) -> np.ndarray:
        lo, hi = self.cfg.noise_snr_range
        snr_db = np.random.uniform(lo, hi)
        rms_signal = np.sqrt(np.mean(audio ** 2)) + 1e-8
        rms_noise = rms_signal / (10.0 ** (snr_db / 20.0))
        noise = np.random.randn(len(audio)).astype(np.float32) * rms_noise
        return (audio + noise).astype(np.float32)

    def _time_mask(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Zero out 1..n short spans (80..max_ms) so the model uses surrounding harmonic context."""
        out = audio.copy()
        max_samples = max(1, int(getattr(self.cfg, "time_mask_max_ms", 150) / 1000.0 * sr))
        min_samples = min(max_samples, max(1, int(0.06 * sr)))
        n_masks = max(1, int(getattr(self.cfg, "n_time_masks", 2)))
        total = len(out)
        if total <= max_samples * 2:
            return out
        for _ in range(np.random.randint(1, n_masks + 1)):
            span = np.random.randint(min_samples, max_samples + 1)
            start = np.random.randint(0, total - span)
            out[start: start + span] = 0.0
        return out


def pitch_shift_root(root_id: int, n_steps: int) -> int:
    if root_id < 0:
        return root_id
    return (root_id + n_steps) % 12


def pitch_shift_bass(bass_id: int, n_steps: int) -> int:
    if bass_id < 0 or bass_id >= 12:
        return bass_id
    return (bass_id + n_steps) % 12
