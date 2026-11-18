from __future__ import annotations

import numpy as np
import torch
import torchaudio.functional as AF

from src.config import AugmentCfg
from src.music_theory import QE_TO_TYPE, ROOT_SILENCE_ID

_MAJOR_SCALE_STEPS = np.array([0, 2, 4, 5, 7, 9, 11], dtype=np.int64)
_MINOR_SCALE_STEPS = np.array([0, 2, 3, 5, 7, 8, 10], dtype=np.int64)


def _synth_piano_like_note(freq_hz: float, n_samples: int, sr: int) -> np.ndarray:
    """Fast vectorized harmonic synthesis (fundamental + 3 overtones + hammer attack & decay)."""
    if n_samples <= 0 or freq_hz <= 20.0:
        return np.zeros(max(0, n_samples), dtype=np.float32)
    t = np.arange(n_samples, dtype=np.float32) / float(sr)
    phase = 2.0 * np.pi * float(freq_hz) * t
    # Piano-like harmonic roll-off with slight random initial phase
    p0 = float(np.random.uniform(0.0, 2.0 * np.pi))
    wave = (
        1.00 * np.sin(phase + p0)
        + 0.42 * np.sin(2.0 * phase + 1.7 * p0) * np.exp(-1.5 * t)
        + 0.18 * np.sin(3.0 * phase + 0.9 * p0) * np.exp(-3.0 * t)
        + 0.08 * np.sin(4.0 * phase) * np.exp(-4.5 * t)
    )
    attack_n = min(n_samples, max(1, int(0.006 * sr)))
    env = np.exp(-2.2 * t).astype(np.float32)
    env[:attack_n] *= np.linspace(0.0, 1.0, attack_n, dtype=np.float32)
    rel_n = min(n_samples // 4, max(1, int(0.012 * sr)))
    if rel_n > 1:
        env[-rel_n:] *= np.linspace(1.0, 0.0, rel_n, dtype=np.float32)
    return (wave * env).astype(np.float32)


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

    def apply_v5_musical_augment(
        self,
        audio: np.ndarray,
        sr: int,
        frame_times_ms: np.ndarray,
        crop_start_ms: float,
        root_arr: np.ndarray,
        qual_arr: np.ndarray,
        ext_arr: np.ndarray,
        type_arr: np.ndarray,
        crf_arr: np.ndarray,
        bass_arr: np.ndarray,
        key_arr: np.ndarray,
        shell_chroma: np.ndarray,
        full_chroma: np.ndarray,
    ) -> np.ndarray:
        """Apply v5 musical augmentations in-place on label arrays and return modified `audio` (`float32`).

        1. `silence_intro_prob`: Inject realistic silent intro/outro (`-65..-52 dBFS`) labelled `root = 12 (N)`.
        2. `melody_overlay_prob`:
           - Fast Passing Melody (`80%`): `130-280ms` upper-register melodic notes over unchanged chord labels.
           - Sustained Tension (`20%`): `>=850ms` sustained 9th over bare `maj`/`min` triad -> upgrades label to `add9`/`madd9`.
        """
        if not self.cfg.enabled:
            return audio

        out = audio.copy()
        total_samples = len(out)

        # 1. Synthetic silent intro or outro to sharpen `N` (no-sound) onset/offset detection
        sil_prob = float(getattr(self.cfg, "silence_intro_prob", 0.0))
        if sil_prob > 0.0 and total_samples >= sr * 4 and np.random.random() < sil_prob:
            sil_sec = float(np.random.uniform(0.6, min(2.2, total_samples / sr * 0.25)))
            sil_samples = int(sil_sec * sr)
            hiss_amp = float(10.0 ** (np.random.uniform(-66.0, -53.0) / 20.0))
            hiss = np.random.randn(sil_samples).astype(np.float32) * hiss_amp
            rel_ft_ms = frame_times_ms - crop_start_ms
            if np.random.random() < 0.7:
                # Silent intro
                out[:sil_samples] = hiss
                fade = min(int(0.015 * sr), total_samples - sil_samples)
                if fade > 1:
                    out[sil_samples: sil_samples + fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
                f_mask = rel_ft_ms < (sil_sec * 1000.0 - 20.0)
            else:
                # Silent outro
                out[-sil_samples:] = hiss
                fade = min(int(0.015 * sr), total_samples - sil_samples)
                if fade > 1:
                    out[-sil_samples - fade: -sil_samples] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
                total_ms = total_samples / float(sr) * 1000.0
                f_mask = rel_ft_ms > (total_ms - sil_sec * 1000.0 + 20.0)

            if f_mask.any():
                root_arr[f_mask] = ROOT_SILENCE_ID
                qual_arr[f_mask] = -100
                ext_arr[f_mask] = -100
                type_arr[f_mask] = -100
                crf_arr[f_mask] = 0
                bass_arr[f_mask] = -100
                shell_chroma[f_mask] = 0.0
                full_chroma[f_mask] = 0.0

        # 2. Melody-Overlay / Sustained Tension Augmentation
        mel_prob = float(getattr(self.cfg, "melody_overlay_prob", 0.0))
        if mel_prob <= 0.0 or np.random.random() >= mel_prob:
            return out

        active_frames = (root_arr >= 0) & (root_arr < 12)
        if active_frames.sum() < 30:
            return out

        rms_bg = float(np.sqrt(np.mean(out ** 2) + 1e-8))
        if rms_bg < 2e-3:
            return out

        sust_prob = float(getattr(self.cfg, "sustained_tension_prob", 0.2))
        applied_sustained = False

        if np.random.random() < sust_prob:
            # Look for a contiguous bare Major (q=0, e=0) or Minor (q=1, e=0) span >= 70 frames (~0.93s)
            bare_mask = active_frames & ((qual_arr == 0) | (qual_arr == 1)) & (ext_arr == 0)
            spans: list[tuple[int, int]] = []
            i = 0
            n_f = len(root_arr)
            while i < n_f:
                if not bare_mask[i]:
                    i += 1
                    continue
                j = i + 1
                while j < n_f and bare_mask[j] and root_arr[j] == root_arr[i] and qual_arr[j] == qual_arr[i]:
                    j += 1
                if (j - i) >= 70:
                    spans.append((i, j))
                i = j

            if spans:
                f_s, f_e = spans[int(np.random.randint(0, len(spans)))]
                r_id = int(root_arr[f_s])
                q_id = int(qual_arr[f_s])
                ninth_pc = (r_id + 2) % 12
                midi_note = 60 + ninth_pc + 12 * int(np.random.randint(0, 2))  # C4..B5
                freq = 440.0 * (2.0 ** ((midi_note - 69.0) / 12.0))

                s_samp = max(0, int((frame_times_ms[f_s] - crop_start_ms) / 1000.0 * sr))
                e_samp = min(total_samples, int((frame_times_ms[f_e - 1] - crop_start_ms) / 1000.0 * sr))
                dur_samp = e_samp - s_samp
                if dur_samp >= int(0.75 * sr):
                    note_wav = _synth_piano_like_note(freq, dur_samp, sr)
                    note_rms = float(np.sqrt(np.mean(note_wav ** 2) + 1e-8))
                    snr_db = float(np.random.uniform(-4.5, 0.5))
                    scale = (rms_bg * (10.0 ** (snr_db / 20.0))) / note_rms
                    out[s_samp:e_samp] += note_wav * scale

                    # Upgrade target chord in-place: F + sustained G -> Fadd9 (skeleton stays identical!)
                    new_tid = int(QE_TO_TYPE[q_id, 6])  # ext=6 is add9 / madd9
                    ext_arr[f_s:f_e] = 6
                    type_arr[f_s:f_e] = new_tid
                    full_chroma[f_s:f_e, ninth_pc] = 1.0
                    applied_sustained = True

        if not applied_sustained:
            # Fast Passing Melody Overlay (120-280ms notes in C4..C6) — chord labels MUST NOT change
            valid_keys = key_arr[(key_arr >= 0) & (key_arr < 24)]
            dom_key = int(np.bincount(valid_keys).argmax()) if len(valid_keys) > 0 else 0
            tonic = dom_key % 12
            scale_steps = _MAJOR_SCALE_STEPS if dom_key < 12 else _MINOR_SCALE_STEPS
            scale_pcs = (tonic + scale_steps) % 12

            melody_track = np.zeros(total_samples, dtype=np.float32)
            pos = int(np.random.uniform(0.0, 0.4) * sr)
            deg_idx = int(np.random.randint(0, 7))
            octave = 5  # MIDI 60..84 (C4..C6)

            while pos < total_samples - int(0.15 * sr):
                f_idx = min(len(root_arr) - 1, int(pos / float(total_samples) * len(root_arr)))
                if root_arr[f_idx] == ROOT_SILENCE_ID or np.random.random() < 0.18:
                    # Rest gap or respect silent region
                    pos += int(np.random.uniform(0.15, 0.35) * sr)
                    continue

                dur_s = float(np.random.uniform(0.12, 0.27))
                n_samp = min(int(dur_s * sr), total_samples - pos)
                # Stepwise contour (+-1 or +-2 scale degrees) + 12% chromatic passing tone
                step = int(np.random.choice([-2, -1, -1, 1, 1, 2]))
                deg_idx = (deg_idx + step) % 7
                if deg_idx == 0 and step > 0 and octave < 6:
                    octave += 1
                elif deg_idx == 6 and step < 0 and octave > 4:
                    octave -= 1

                pc = int(scale_pcs[deg_idx])
                if np.random.random() < 0.12:
                    pc = (pc + 1) % 12
                midi_note = 12 * octave + pc
                freq = 440.0 * (2.0 ** ((midi_note - 69.0) / 12.0))
                melody_track[pos: pos + n_samp] += _synth_piano_like_note(freq, n_samp, sr)
                pos += n_samp

            mel_rms = float(np.sqrt(np.mean(melody_track ** 2) + 1e-8))
            if mel_rms > 1e-6:
                snr_db = float(np.random.uniform(-7.0, 1.0))
                scale = (rms_bg * (10.0 ** (snr_db / 20.0))) / mel_rms
                out = out + melody_track * scale

        peak = float(np.max(np.abs(out))) if len(out) else 0.0
        if peak > 0.99:
            out = out * (0.99 / peak)
        return out.astype(np.float32)

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
    if root_id < 0 or root_id >= 12:
        return root_id
    return (root_id + n_steps) % 12


def pitch_shift_bass(bass_id: int, n_steps: int) -> int:
    if bass_id < 0 or bass_id >= 12:
        return bass_id
    return (bass_id + n_steps) % 12
