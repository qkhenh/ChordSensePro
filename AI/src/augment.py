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
            pitch_shift_n = np.random.randint(lo, hi + 1)
            if pitch_shift_n != 0:
                audio = self._pitch_shift(audio, sr, pitch_shift_n)

            audio = self._gain(audio)

            if np.random.random() < self.cfg.noise_prob:
                audio = self._add_noise(audio)

        return audio, pitch_shift_n

    def _pitch_shift(self, audio: np.ndarray, sr: int, n_steps: int) -> np.ndarray:
        wav_t = torch.from_numpy(audio).unsqueeze(0)
        shifted = AF.pitch_shift(wav_t, sr, n_steps)
        return shifted.squeeze(0).numpy()

    def _gain(self, audio: np.ndarray) -> np.ndarray:
        gain_db = np.random.uniform(-self.cfg.gain_db, self.cfg.gain_db)
        return audio * (10.0 ** (gain_db / 20.0))

    def _add_noise(self, audio: np.ndarray) -> np.ndarray:
        lo, hi = self.cfg.noise_snr_range
        snr_db = np.random.uniform(lo, hi)
        rms_signal = np.sqrt(np.mean(audio ** 2)) + 1e-8
        rms_noise = rms_signal / (10.0 ** (snr_db / 20.0))
        noise = np.random.randn(len(audio)).astype(np.float32) * rms_noise
        return audio + noise


def pitch_shift_root(root_id: int, n_steps: int) -> int:
    if root_id < 0:
        return root_id
    return (root_id + n_steps) % 12


def pitch_shift_bass(bass_id: int, n_steps: int) -> int:
    if bass_id < 0 or bass_id >= 12:
        return bass_id
    return (bass_id + n_steps) % 12


