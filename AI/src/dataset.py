from __future__ import annotations
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.augment import apply_cqt_spec_augment, pitch_shift_bass, pitch_shift_root
from src.config import Config
from src.model import (
    NORM_STD_FLOOR,
    compute_chunked_frame_rms_db,
    mert_chunked_frame_times_ms,
    mert_frame_times_ms,
    mert_output_length,
)
from src.music_theory import (
    BASS_NONE_ID,
    N_SKELETONS_V6,
    N_TENSIONS_V6,
    ROOT_SILENCE_ID,
    SKEL14_BASE_QE,
    chord_chroma_targets,
    chord_chroma_targets_v6,
    encode_crf_skeleton_state,
    encode_crf_skeleton_state_v6,
    estimate_local_keys,
    parse_v6_label_targets,
)


def normalize_chunks_clamped(wav_float: np.ndarray, chunk_samples: int) -> np.ndarray:
    """Normalize each `chunk_samples` block with a `-55 dBFS` std floor so silence never explodes."""
    out = np.empty_like(wav_float, dtype=np.float32)
    total = len(wav_float)
    pos = 0
    while pos < total:
        end = min(total, pos + chunk_samples)
        seg = wav_float[pos:end]
        denom = max(float(seg.std()), NORM_STD_FLOOR)
        out[pos:end] = (seg - float(seg.mean())) / denom
        pos = end
    return out


class ChordDataset(Dataset):

    def __init__(self, cfg: Config, split: str = "train", augment_fn=None):
        self.cfg = cfg
        self.split = split
        self.augment_fn = augment_fn if split == "train" else None
        self.crop_samples = cfg.crop_samples
        self.chunk_samples = cfg.chunk_samples
        self.sr = cfg.sample_rate
        self.is_v5 = getattr(cfg.model, "arch_version", 4) >= 5
        self.is_v6 = getattr(cfg.model, "arch_version", 4) >= 6

        split_dir = Path(cfg.data_dir) / split
        chords_df = pd.read_parquet(split_dir / "chords.parquet")

        # Anti-Overfitting Guardrail for Synthetic Rare Data:
        # Phase 2 (LoRA): strictly 0% synthetic rare songs!
        # Phase 1 & 3: track synthetic song IDs to cap <= cfg.train.max_synthetic_ratio (default 4%).
        self._synthetic_songs: set[str] = set()
        if "is_synthetic_rare" in chords_df.columns:
            syn_mask = chords_df["is_synthetic_rare"].fillna(False).astype(bool)
            syn_sids = set(chords_df.loc[syn_mask, "song_id"].unique())
            if self.is_v6 and split == "train" and int(getattr(cfg.train, "phase", 1)) == 2:
                chords_df = chords_df.loc[~syn_mask].reset_index(drop=True)
            else:
                self._synthetic_songs = syn_sids

        self.audio_dir = split_dir / "audio"
        self.song_ids = sorted(chords_df["song_id"].unique())
        self.song_to_idx = {sid: i for i, sid in enumerate(self.song_ids)}

        self._chords_by_song: dict[str, pd.DataFrame] = {}
        self._fast_chords: dict[str, tuple[np.ndarray, ...]] = {}
        self._v5_chords: dict[str, dict[str, np.ndarray]] = {}

        for sid, group in chords_df.groupby("song_id"):
            g = group.sort_values("start_ms").reset_index(drop=True)
            self._chords_by_song[sid] = g

            starts = g["start_ms"].to_numpy(dtype=np.float64)
            ends = g["end_ms"].to_numpy(dtype=np.float64)
            roots = np.where(g["root_id"].notna(), g["root_id"].fillna(-100).astype(np.int64), -100)
            quals = np.where(g["quality_id"].notna(), g["quality_id"].fillna(-100).astype(np.int64), -100)
            exts = np.where(g["ext_id"].notna(), g["ext_id"].fillna(-100).astype(np.int64), -100)
            if "bass_id" in g.columns:
                basses = np.where(
                    g["bass_id"].notna(),
                    g["bass_id"].fillna(-100).astype(np.int64),
                    np.where(roots == -100, -100, 12),
                )
            else:
                basses = np.where(roots == -100, -100, 12)
            self._fast_chords[sid] = (starts, ends, roots, quals, exts, basses)

            if self.is_v6:
                n_ev = len(starts)
                skel14s = np.full(n_ev, -100, dtype=np.int64)
                tension_bits = np.zeros((n_ev, N_TENSIONS_V6), dtype=np.float32)
                tension_valid = np.zeros(n_ev, dtype=bool)
                inversions = np.full(n_ev, -100, dtype=np.int64)
                crfs = np.full(n_ev, -100, dtype=np.int64)
                shell_c = np.zeros((n_ev, 12), dtype=np.float32)
                full_c = np.zeros((n_ev, 12), dtype=np.float32)
                durs_ms = np.maximum(0.0, ends - starts).astype(np.float32)

                has_v4_cols = {"skel14_id", "tension_mask", "tension_valid", "inversion_id"} <= set(g.columns)
                if has_v4_cols:
                    s14_col = g["skel14_id"].fillna(-1).astype(np.int64).to_numpy()
                    tm_col = g["tension_mask"].fillna(0).astype(np.int64).to_numpy()
                    tv_col = g["tension_valid"].fillna(False).astype(bool).to_numpy()
                    inv_col = g["inversion_id"].fillna(-1).astype(np.int64).to_numpy()
                    for k in range(n_ev):
                        r_k, sk_k, tm_k, b_k = int(roots[k]), int(s14_col[k]), int(tm_col[k]), int(basses[k])
                        skel14s[k] = sk_k if 0 <= sk_k < N_SKELETONS_V6 else -100
                        tension_valid[k] = bool(tv_col[k])
                        inversions[k] = int(inv_col[k]) if 0 <= int(inv_col[k]) < 5 else -100
                        for bi in range(N_TENSIONS_V6):
                            if tm_k & (1 << bi):
                                tension_bits[k, bi] = 1.0
                        crfs[k] = encode_crf_skeleton_state_v6(r_k, sk_k)
                        sh_m, fu_m = chord_chroma_targets_v6(r_k, sk_k, tm_k, b_k)
                        shell_c[k] = sh_m
                        full_c[k] = fu_m
                else:
                    raw_col = g["label_raw"].tolist() if "label_raw" in g.columns else [None] * n_ev
                    for k in range(n_ev):
                        r_k, q_k, e_k, b_k = int(roots[k]), int(quals[k]), int(exts[k]), int(basses[k])
                        raw_s = str(raw_col[k]) if raw_col[k] is not None and pd.notna(raw_col[k]) else None
                        t6 = parse_v6_label_targets(raw_s, r_k if r_k >= 0 else None, q_k if q_k >= 0 else None, e_k if e_k >= 0 else None, b_k if b_k >= 0 else BASS_NONE_ID)
                        sk_k = int(t6["skel14_id"])
                        tm_k = int(t6["tension_mask"])
                        inv_k = int(t6["inversion_id"])
                        skel14s[k] = sk_k if 0 <= sk_k < N_SKELETONS_V6 else -100
                        tension_valid[k] = bool(t6["tension_valid"])
                        inversions[k] = inv_k if 0 <= inv_k < 5 else -100
                        for bi in range(N_TENSIONS_V6):
                            if tm_k & (1 << bi):
                                tension_bits[k, bi] = 1.0
                        crfs[k] = encode_crf_skeleton_state_v6(r_k, sk_k)
                        sh_m, fu_m = chord_chroma_targets_v6(r_k, sk_k, tm_k, b_k)
                        shell_c[k] = sh_m
                        full_c[k] = fu_m

                local_keys = estimate_local_keys(starts, ends, roots, quals, exts)
                true_basses = np.where(
                    (basses >= 0) & (basses < 12),
                    basses,
                    np.where((roots >= 0) & (roots < 12), roots, -100),
                ).astype(np.int64)
                self._v5_chords[sid] = {
                    "starts": starts,
                    "ends": ends,
                    "durs_ms": durs_ms,
                    "roots": roots,
                    "quals": quals,
                    "exts": exts,
                    "basses": basses,
                    "true_basses": true_basses,
                    "skel14s": skel14s,
                    "tension_bits": tension_bits,
                    "tension_valid": tension_valid,
                    "inversions": inversions,
                    "crfs": crfs,
                    "keys": local_keys,
                    "shell_c": shell_c,
                    "full_c": full_c,
                }
            elif self.is_v5:
                n_ev = len(starts)
                types = np.full(n_ev, -100, dtype=np.int64)
                skels = np.full(n_ev, -100, dtype=np.int64)
                crfs = np.full(n_ev, -100, dtype=np.int64)
                shell_c = np.zeros((n_ev, 12), dtype=np.float32)
                full_c = np.zeros((n_ev, 12), dtype=np.float32)
                raw_col = g["label_raw"].tolist() if "label_raw" in g.columns else [None] * n_ev

                for k in range(n_ev):
                    r_k, q_k, e_k, b_k = int(roots[k]), int(quals[k]), int(exts[k]), int(basses[k])
                    tid, sk_id, crf_id = encode_crf_skeleton_state(r_k, q_k, e_k)
                    types[k] = tid
                    skels[k] = sk_id
                    crfs[k] = crf_id
                    raw_s = str(raw_col[k]) if raw_col[k] is not None and pd.notna(raw_col[k]) else None
                    sh_m, fu_m = chord_chroma_targets(r_k, q_k, e_k, b_k, label_raw=raw_s)
                    shell_c[k] = sh_m
                    full_c[k] = fu_m

                local_keys = estimate_local_keys(starts, ends, roots, quals, exts)
                true_basses = np.where(
                    (basses >= 0) & (basses < 12),
                    basses,
                    np.where((roots >= 0) & (roots < 12), roots, -100),
                ).astype(np.int64)
                self._v5_chords[sid] = {
                    "starts": starts,
                    "ends": ends,
                    "roots": roots,
                    "quals": quals,
                    "exts": exts,
                    "basses": basses,
                    "true_basses": true_basses,
                    "types": types,
                    "skels": skels,
                    "crfs": crfs,
                    "keys": local_keys,
                    "shell_c": shell_c,
                    "full_c": full_c,
                }

        self._audio_lengths: dict[str, int] = {}
        for sid in self.song_ids:
            npy_path = self.audio_dir / f"{sid}.npy"
            header = np.lib.format.open_memmap(str(npy_path), mode="r")
            self._audio_lengths[sid] = len(header)
            del header

        min_required = self.chunk_samples if self.is_v5 else self.crop_samples
        self._valid_songs = [
            sid for sid in self.song_ids
            if self._audio_lengths[sid] >= min_required
        ]

        # Precompute crop bounds [lo_sample, hi_sample] per song & sample index list
        self._crop_bounds: dict[str, tuple[int, int]] = {}
        self._samples: list[tuple[str, int | None, float | None]] = []

        for sid in self._valid_songs:
            audio_len = self._audio_lengths[sid]
            if self.is_v5:
                # v5 covers full song [0, max_start] (including silent intros/outros for `N` learning)
                max_start = max(0, audio_len - self.crop_samples)
                lo, hi = 0, max_start
                dur_sec = audio_len / float(self.sr)
                if self.split == "train":
                    n_crops = min(6, max(1, int(round(dur_sec / max(float(self.cfg.crop_s), 15.0)))))
                    for _ in range(n_crops):
                        self._samples.append((sid, None, None))
                else:
                    tiles: list[tuple[int, float | None]] = []
                    if audio_len <= self.crop_samples:
                        tiles.append((0, None))
                    else:
                        pos = 0
                        prev_end_ms: float | None = None
                        while pos <= hi:
                            tiles.append((pos, prev_end_ms))
                            prev_end_ms = (pos + self.crop_samples) / float(self.sr) * 1000.0
                            if pos == hi:
                                break
                            if pos + self.crop_samples > hi:
                                tiles.append((hi, prev_end_ms))
                                break
                            pos += self.crop_samples

                    max_val_tiles = 4
                    if self.split == "val" and len(tiles) > max_val_tiles:
                        idxs = np.linspace(0, len(tiles) - 1, max_val_tiles, dtype=int)
                        for ti in idxs:
                            self._samples.append((sid, tiles[int(ti)][0], None))
                    else:
                        for t_pos, t_mask in tiles:
                            self._samples.append((sid, t_pos, t_mask))
            else:
                max_start = audio_len - self.crop_samples
                fc = self._fast_chords.get(sid)
                if fc is not None and len(fc[0]) > 0:
                    first_ms = float(fc[0][0])
                    last_ms = float(fc[1][-1])
                    start_s = int(first_ms / 1000.0 * self.sr)
                    end_minus_crop = int(last_ms / 1000.0 * self.sr) - self.crop_samples
                    lo = max(0, min(max_start, min(start_s, end_minus_crop)))
                    hi = max(lo, min(max_start, max(start_s, end_minus_crop)))
                    labelled_sec = max(0.0, (last_ms - first_ms) / 1000.0)
                else:
                    lo, hi = 0, max_start
                    labelled_sec = audio_len / float(self.sr)

                self._crop_bounds[sid] = (lo, hi)

                if self.split == "train":
                    n_crops = min(6, max(1, int(round(labelled_sec / 15.0))))
                    for _ in range(n_crops):
                        self._samples.append((sid, None, None))
                else:
                    tiles = []
                    pos = lo
                    prev_end_ms = None
                    while pos <= hi:
                        tiles.append((pos, prev_end_ms))
                        prev_end_ms = (pos + self.crop_samples) / float(self.sr) * 1000.0
                        if pos == hi:
                            break
                        if pos + self.crop_samples > hi:
                            tiles.append((hi, prev_end_ms))
                            break
                        pos += self.crop_samples

                    max_val_tiles = 6
                    if self.split == "val" and len(tiles) > max_val_tiles:
                        idxs = np.linspace(0, len(tiles) - 1, max_val_tiles, dtype=int)
                        for ti in idxs:
                            self._samples.append((sid, tiles[int(ti)][0], None))
                    else:
                        for t_pos, t_mask in tiles:
                            self._samples.append((sid, t_pos, t_mask))

            if sid not in self._crop_bounds:
                self._crop_bounds[sid] = (lo, hi)

        # Anti-overfitting guardrail: cap synthetic rare-chord items to <= max_synthetic_ratio in Phase 1/3
        if self.is_v6 and self.split == "train" and self._synthetic_songs:
            max_ratio = float(getattr(cfg.train, "max_synthetic_ratio", 0.04))
            real_items = [s for s in self._samples if s[0] not in self._synthetic_songs]
            syn_items = [s for s in self._samples if s[0] in self._synthetic_songs]
            if real_items and syn_items:
                max_syn = int(math.floor(len(real_items) * max_ratio / max(1.0 - max_ratio, 1e-6)))
                if max_syn <= 0:
                    self._samples = real_items
                elif len(syn_items) > max_syn:
                    rng = np.random.default_rng(int(getattr(cfg.train, "seed", 42)))
                    keep = rng.choice(len(syn_items), size=max_syn, replace=False)
                    self._samples = real_items + [syn_items[int(i)] for i in sorted(keep)]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        if self.is_v6:
            return self._getitem_v6(idx)
        if self.is_v5:
            return self._getitem_v5(idx)

        song_id, fixed_start, min_eval_ms = self._samples[idx]
        lo, hi = self._crop_bounds[song_id]

        if fixed_start is None:
            crop_start = int(np.random.randint(lo, hi + 1))
        else:
            crop_start = fixed_start

        audio = np.load(self.audio_dir / f"{song_id}.npy", mmap_mode="r")
        crop = audio[crop_start: crop_start + self.crop_samples].astype(np.float32) / 32768.0

        crop_start_ms = crop_start / self.sr * 1000.0

        pitch_shift_n = 0
        if self.augment_fn is not None:
            crop, pitch_shift_n = self.augment_fn(crop, self.sr)

        std = crop.std()
        if std > 1e-6:
            crop = (crop - crop.mean()) / std

        if self.cfg.model.frame_level:
            n_frames = mert_output_length(self.crop_samples)
            root_arr, qual_arr, ext_arr, bass_arr = self._lookup_chord_frames(
                song_id, crop_start_ms, n_frames, min_eval_ms=min_eval_ms
            )
            if pitch_shift_n != 0:
                valid_r = root_arr >= 0
                root_arr = root_arr.copy()
                root_arr[valid_r] = (root_arr[valid_r] + pitch_shift_n) % 12
                valid_b = (bass_arr >= 0) & (bass_arr < 12)
                bass_arr = bass_arr.copy()
                bass_arr[valid_b] = (bass_arr[valid_b] + pitch_shift_n) % 12
            return {
                "wav": torch.from_numpy(crop),
                "root": torch.from_numpy(root_arr),
                "quality": torch.from_numpy(qual_arr),
                "ext": torch.from_numpy(ext_arr),
                "bass": torch.from_numpy(bass_arr),
                "song_id": song_id,
            }

        mid_ms = crop_start_ms + (self.cfg.crop_s * 500)
        root_id, quality_id, ext_id, bass_id = self._lookup_chord(song_id, mid_ms)

        if pitch_shift_n != 0:
            root_id = pitch_shift_root(root_id, pitch_shift_n)
            bass_id = pitch_shift_bass(bass_id, pitch_shift_n)

        return {
            "wav": torch.from_numpy(crop),
            "root": torch.tensor(root_id, dtype=torch.long),
            "quality": torch.tensor(quality_id, dtype=torch.long),
            "ext": torch.tensor(ext_id, dtype=torch.long),
            "bass": torch.tensor(bass_id, dtype=torch.long),
            "song_id": song_id,
        }

    def _getitem_v5(self, idx: int) -> dict:
        song_id, fixed_start, min_eval_ms = self._samples[idx]
        audio_len = self._audio_lengths[song_id]
        lo, hi = self._crop_bounds[song_id]

        audio = np.load(self.audio_dir / f"{song_id}.npy", mmap_mode="r")
        n_chunks = max(1, self.crop_samples // self.chunk_samples)
        total_samples = n_chunks * self.chunk_samples

        if audio_len >= total_samples:
            crop_start = int(np.random.randint(lo, hi + 1)) if fixed_start is None else int(fixed_start)
            raw_crop = audio[crop_start: crop_start + total_samples].astype(np.float32) / 32768.0
            pad_offset = 0
            valid_audio_end_ms = (crop_start + total_samples) / float(self.sr) * 1000.0
        else:
            # Song is shorter than 20s window (e.g. 14s Hooktheory clip): place inside 20s buffer with quiet room floor
            crop_start = 0
            spare = total_samples - audio_len
            pad_offset = int(np.random.randint(0, spare + 1)) if (self.split == "train" and fixed_start is None) else 0
            raw_crop = np.random.randn(total_samples).astype(np.float32) * 3e-4  # ~-70 dBFS room floor
            raw_crop[pad_offset: pad_offset + audio_len] = audio[:audio_len].astype(np.float32) / 32768.0
            # Effective start_ms relative to buffer index 0 can be negative when pad_offset > 0
            crop_start = -pad_offset
            valid_audio_end_ms = audio_len / float(self.sr) * 1000.0

        crop_start_ms = crop_start / float(self.sr) * 1000.0
        frame_times = mert_chunked_frame_times_ms(crop_start_ms, n_chunks, self.chunk_samples, self.sr)

        labels = self._lookup_v5_frames(
            song_id,
            frame_times,
            min_eval_ms=min_eval_ms,
            max_eval_ms=(valid_audio_end_ms if self.split != "train" else None),
        )

        if self.augment_fn is not None:
            raw_crop, pitch_shift_n = self.augment_fn(raw_crop, self.sr)
            if pitch_shift_n != 0:
                vr = (labels["root"] >= 0) & (labels["root"] < 12)
                labels["root"][vr] = (labels["root"][vr] + pitch_shift_n) % 12
                vb = (labels["bass"] >= 0) & (labels["bass"] < 12)
                labels["bass"][vb] = (labels["bass"][vb] + pitch_shift_n) % 12
                vtb = (labels["true_bass"] >= 0) & (labels["true_bass"] < 12)
                labels["true_bass"][vtb] = (labels["true_bass"][vtb] + pitch_shift_n) % 12
                vc_mask = vr & (labels["skel_id"] >= 0)
                labels["crf_state"][vc_mask] = 1 + labels["root"][vc_mask] * 12 + labels["skel_id"][vc_mask]
                vk = (labels["local_key"] >= 0) & (labels["local_key"] < 24)
                is_min_k = labels["local_key"][vk] >= 12
                labels["local_key"][vk] = np.where(
                    is_min_k,
                    12 + ((labels["local_key"][vk] - 12 + pitch_shift_n) % 12),
                    (labels["local_key"][vk] + pitch_shift_n) % 12,
                )
                labels["shell_chroma"] = np.roll(labels["shell_chroma"], pitch_shift_n, axis=-1)
                labels["full_chroma"] = np.roll(labels["full_chroma"], pitch_shift_n, axis=-1)
            if hasattr(self.augment_fn, "apply_v5_musical_augment"):
                raw_crop = self.augment_fn.apply_v5_musical_augment(
                    raw_crop,
                    self.sr,
                    frame_times,
                    crop_start_ms,
                    labels["root"],
                    labels["quality"],
                    labels["ext"],
                    labels["type_id"],
                    labels["crf_state"],
                    labels["bass"],
                    labels["local_key"],
                    labels["shell_chroma"],
                    labels["full_chroma"],
                    true_bass_arr=labels["true_bass"],
                )

        # Compute pre-normalization dBFS per frame & assign `N` (`root = 12`) to true silent frames
        frame_rms_db = compute_chunked_frame_rms_db(raw_crop, n_chunks, self.chunk_samples)
        sil_thresh = float(self.cfg.model.silence_db_threshold)
        in_eval_window = (frame_times >= 0.0) & (frame_times <= valid_audio_end_ms)
        if self.split == "train":
            # In train, both padded buffer regions and true silent frames (< -45 dBFS) teach `N`
            silent_frames = frame_rms_db < sil_thresh
        else:
            # In val/test, only mark silent frames inside the real song audio bounds (never evaluate padding)
            silent_frames = in_eval_window & (frame_rms_db < (sil_thresh - 3.0)) & (labels["root"] == -100)

        if silent_frames.any():
            labels["root"][silent_frames] = ROOT_SILENCE_ID
            labels["quality"][silent_frames] = -100
            labels["ext"][silent_frames] = -100
            labels["type_id"][silent_frames] = -100
            labels["crf_state"][silent_frames] = 0
            labels["bass"][silent_frames] = -100
            labels["true_bass"][silent_frames] = ROOT_SILENCE_ID
            labels["shell_chroma"][silent_frames] = 0.0
            labels["full_chroma"][silent_frames] = 0.0

        norm_crop = normalize_chunks_clamped(raw_crop, self.chunk_samples)

        return {
            "wav": torch.from_numpy(norm_crop),
            "frame_rms_db": torch.from_numpy(frame_rms_db),
            "root": torch.from_numpy(labels["root"]),
            "quality": torch.from_numpy(labels["quality"]),
            "ext": torch.from_numpy(labels["ext"]),
            "type_id": torch.from_numpy(labels["type_id"]),
            "skel_id": torch.from_numpy(labels["skel_id"]),
            "crf_state": torch.from_numpy(labels["crf_state"]),
            "bass": torch.from_numpy(labels["bass"]),
            "true_bass": torch.from_numpy(labels["true_bass"]),
            "local_key": torch.from_numpy(labels["local_key"]),
            "shell_chroma": torch.from_numpy(labels["shell_chroma"]),
            "full_chroma": torch.from_numpy(labels["full_chroma"]),
            "song_id": song_id,
        }

    @staticmethod
    def _compute_gaussian_boundaries(
        root_arr: np.ndarray,
        skel_arr: np.ndarray,
        true_bass_arr: np.ndarray,
        sigma_frames: float = 2.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build Gaussian-smoothed `[0, 1]` chord-change and bass-change boundary targets (`sigma = 26.7ms`)."""
        n = len(root_arr)
        bnd_chord = np.zeros(n, dtype=np.float32)
        bnd_bass = np.zeros(n, dtype=np.float32)
        if n <= 1:
            return bnd_chord, bnd_bass

        rad = max(1, int(round(3.0 * sigma_frames)))
        offsets = np.arange(-rad, rad + 1, dtype=np.float32)
        gauss = np.exp(-0.5 * (offsets / float(sigma_frames)) ** 2).astype(np.float32)

        def _stamp(target: np.ndarray, pos_list: np.ndarray) -> None:
            for p in pos_list:
                lo = max(0, int(p) - rad)
                hi = min(n, int(p) + rad + 1)
                g_lo = lo - (int(p) - rad)
                g_hi = g_lo + (hi - lo)
                target[lo:hi] = np.maximum(target[lo:hi], gauss[g_lo:g_hi])

        obs_c = (root_arr[:-1] >= 0) & (root_arr[1:] >= 0)
        chg_c = obs_c & ((root_arr[1:] != root_arr[:-1]) | (skel_arr[1:] != skel_arr[:-1]))
        _stamp(bnd_chord, np.nonzero(chg_c)[0] + 1)

        obs_b = (true_bass_arr[:-1] >= 0) & (true_bass_arr[:-1] < 12) & (true_bass_arr[1:] >= 0) & (true_bass_arr[1:] < 12)
        chg_b = obs_b & (true_bass_arr[1:] != true_bass_arr[:-1])
        _stamp(bnd_bass, np.nonzero(chg_b)[0] + 1)
        return bnd_chord, bnd_bass

    def _lookup_v6_frames(
        self,
        song_id: str,
        frame_times: np.ndarray,
        min_eval_ms: float | None = None,
        max_eval_ms: float | None = None,
    ) -> dict[str, np.ndarray]:
        n_frames = len(frame_times)
        out = {
            "root": np.full(n_frames, -100, dtype=np.int64),
            "quality": np.full(n_frames, -100, dtype=np.int64),
            "ext": np.full(n_frames, -100, dtype=np.int64),
            "skel14": np.full(n_frames, -100, dtype=np.int64),
            "tension_bits": np.zeros((n_frames, N_TENSIONS_V6), dtype=np.float32),
            "tension_valid": np.zeros(n_frames, dtype=bool),
            "inversion": np.full(n_frames, -100, dtype=np.int64),
            "crf_state": np.full(n_frames, -100, dtype=np.int64),
            "bass": np.full(n_frames, -100, dtype=np.int64),
            "true_bass": np.full(n_frames, -100, dtype=np.int64),
            "local_key": np.full(n_frames, -100, dtype=np.int64),
            "shell_chroma": np.zeros((n_frames, 12), dtype=np.float32),
            "full_chroma": np.zeros((n_frames, 12), dtype=np.float32),
            "chord_dur_ms": np.zeros(n_frames, dtype=np.float32),
        }
        vc = self._v5_chords.get(song_id)
        if vc is None or len(vc["starts"]) == 0 or n_frames == 0:
            return out

        starts, ends = vc["starts"], vc["ends"]
        idx = np.searchsorted(starts, frame_times, side="right") - 1
        safe_idx = np.maximum(idx, 0)
        valid = (frame_times >= 0.0) & (idx >= 0) & (frame_times < ends[safe_idx])
        if min_eval_ms is not None:
            valid &= frame_times >= min_eval_ms
        if max_eval_ms is not None:
            valid &= frame_times <= max_eval_ms

        if valid.any():
            hit = safe_idx[valid]
            out["root"][valid] = vc["roots"][hit]
            out["quality"][valid] = vc["quals"][hit]
            out["ext"][valid] = vc["exts"][hit]
            out["skel14"][valid] = vc["skel14s"][hit]
            out["tension_bits"][valid] = vc["tension_bits"][hit]
            out["tension_valid"][valid] = vc["tension_valid"][hit]
            out["inversion"][valid] = vc["inversions"][hit]
            out["crf_state"][valid] = vc["crfs"][hit]
            out["bass"][valid] = vc["basses"][hit]
            out["true_bass"][valid] = vc["true_basses"][hit]
            out["local_key"][valid] = vc["keys"][hit]
            out["shell_chroma"][valid] = vc["shell_c"][hit]
            out["full_chroma"][valid] = vc["full_c"][hit]
            out["chord_dur_ms"][valid] = vc["durs_ms"][hit]

        return out

    def _getitem_v6(self, idx: int) -> dict:
        song_id, fixed_start, min_eval_ms = self._samples[idx]
        audio_len = self._audio_lengths[song_id]
        lo, hi = self._crop_bounds[song_id]

        audio = np.load(self.audio_dir / f"{song_id}.npy", mmap_mode="r")
        n_chunks = max(1, self.crop_samples // self.chunk_samples)
        total_samples = n_chunks * self.chunk_samples

        if audio_len >= total_samples:
            crop_start = int(np.random.randint(lo, hi + 1)) if fixed_start is None else int(fixed_start)
            raw_crop = audio[crop_start : crop_start + total_samples].astype(np.float32) / 32768.0
            valid_audio_end_ms = (crop_start + total_samples) / float(self.sr) * 1000.0
        else:
            spare = total_samples - audio_len
            pad_offset = int(np.random.randint(0, spare + 1)) if (self.split == "train" and fixed_start is None) else 0
            raw_crop = np.random.randn(total_samples).astype(np.float32) * 3e-4
            raw_crop[pad_offset : pad_offset + audio_len] = audio[:audio_len].astype(np.float32) / 32768.0
            crop_start = -pad_offset
            valid_audio_end_ms = audio_len / float(self.sr) * 1000.0

        crop_start_ms = crop_start / float(self.sr) * 1000.0
        frame_times = mert_chunked_frame_times_ms(crop_start_ms, n_chunks, self.chunk_samples, self.sr)

        labels = self._lookup_v6_frames(
            song_id,
            frame_times,
            min_eval_ms=min_eval_ms,
            max_eval_ms=(valid_audio_end_ms if self.split != "train" else None),
        )

        if self.augment_fn is not None:
            raw_crop, pitch_shift_n = self.augment_fn(raw_crop, self.sr)
            if pitch_shift_n != 0:
                vr = (labels["root"] >= 0) & (labels["root"] < 12)
                labels["root"][vr] = (labels["root"][vr] + pitch_shift_n) % 12
                vb = (labels["bass"] >= 0) & (labels["bass"] < 12)
                labels["bass"][vb] = (labels["bass"][vb] + pitch_shift_n) % 12
                vtb = (labels["true_bass"] >= 0) & (labels["true_bass"] < 12)
                labels["true_bass"][vtb] = (labels["true_bass"][vtb] + pitch_shift_n) % 12
                vc_mask = vr & (labels["skel14"] >= 0) & (labels["skel14"] < N_SKELETONS_V6)
                labels["crf_state"][vc_mask] = 1 + labels["root"][vc_mask] * N_SKELETONS_V6 + labels["skel14"][vc_mask]
                vk = (labels["local_key"] >= 0) & (labels["local_key"] < 24)
                is_min_k = labels["local_key"][vk] >= 12
                labels["local_key"][vk] = np.where(
                    is_min_k,
                    12 + ((labels["local_key"][vk] - 12 + pitch_shift_n) % 12),
                    (labels["local_key"][vk] + pitch_shift_n) % 12,
                )
                labels["shell_chroma"] = np.roll(labels["shell_chroma"], pitch_shift_n, axis=-1)
                labels["full_chroma"] = np.roll(labels["full_chroma"], pitch_shift_n, axis=-1)
            if hasattr(self.augment_fn, "apply_v6_musical_augment"):
                raw_crop = self.augment_fn.apply_v6_musical_augment(
                    raw_crop, self.sr, frame_times, crop_start_ms, labels
                )

        frame_rms_db = compute_chunked_frame_rms_db(raw_crop, n_chunks, self.chunk_samples)
        sil_thresh = float(self.cfg.model.silence_db_threshold)
        in_eval_window = (frame_times >= 0.0) & (frame_times <= valid_audio_end_ms)
        if self.split == "train":
            silent_frames = frame_rms_db < sil_thresh
        else:
            silent_frames = in_eval_window & (frame_rms_db < (sil_thresh - 3.0)) & (labels["root"] == -100)

        if silent_frames.any():
            labels["root"][silent_frames] = ROOT_SILENCE_ID
            labels["quality"][silent_frames] = -100
            labels["ext"][silent_frames] = -100
            labels["skel14"][silent_frames] = -100
            labels["tension_bits"][silent_frames] = 0.0
            labels["tension_valid"][silent_frames] = False
            labels["inversion"][silent_frames] = -100
            labels["crf_state"][silent_frames] = 0
            labels["bass"][silent_frames] = -100
            labels["true_bass"][silent_frames] = ROOT_SILENCE_ID
            labels["shell_chroma"][silent_frames] = 0.0
            labels["full_chroma"][silent_frames] = 0.0
            labels["chord_dur_ms"][silent_frames] = 0.0

        bnd_chord, bnd_bass = self._compute_gaussian_boundaries(
            labels["root"], labels["skel14"], labels["true_bass"]
        )
        norm_crop = normalize_chunks_clamped(raw_crop, self.chunk_samples)

        return {
            "wav": torch.from_numpy(norm_crop),
            "frame_rms_db": torch.from_numpy(frame_rms_db),
            "root": torch.from_numpy(labels["root"]),
            "quality": torch.from_numpy(labels["quality"]),
            "ext": torch.from_numpy(labels["ext"]),
            "skel14": torch.from_numpy(labels["skel14"]),
            "tension_bits": torch.from_numpy(labels["tension_bits"]),
            "tension_valid": torch.from_numpy(labels["tension_valid"]),
            "inversion": torch.from_numpy(labels["inversion"]),
            "crf_state": torch.from_numpy(labels["crf_state"]),
            "bass": torch.from_numpy(labels["bass"]),
            "true_bass": torch.from_numpy(labels["true_bass"]),
            "local_key": torch.from_numpy(labels["local_key"]),
            "shell_chroma": torch.from_numpy(labels["shell_chroma"]),
            "full_chroma": torch.from_numpy(labels["full_chroma"]),
            "boundary_chord": torch.from_numpy(bnd_chord),
            "boundary_bass": torch.from_numpy(bnd_bass),
            "chord_dur_ms": torch.from_numpy(labels["chord_dur_ms"]),
            "song_id": song_id,
        }

    def _lookup_v5_frames(
        self,
        song_id: str,
        frame_times: np.ndarray,
        min_eval_ms: float | None = None,
        max_eval_ms: float | None = None,
    ) -> dict[str, np.ndarray]:
        n_frames = len(frame_times)
        out = {
            "root": np.full(n_frames, -100, dtype=np.int64),
            "quality": np.full(n_frames, -100, dtype=np.int64),
            "ext": np.full(n_frames, -100, dtype=np.int64),
            "type_id": np.full(n_frames, -100, dtype=np.int64),
            "skel_id": np.full(n_frames, -100, dtype=np.int64),
            "crf_state": np.full(n_frames, -100, dtype=np.int64),
            "bass": np.full(n_frames, -100, dtype=np.int64),
            "true_bass": np.full(n_frames, -100, dtype=np.int64),
            "local_key": np.full(n_frames, -100, dtype=np.int64),
            "shell_chroma": np.zeros((n_frames, 12), dtype=np.float32),
            "full_chroma": np.zeros((n_frames, 12), dtype=np.float32),
        }
        vc = self._v5_chords.get(song_id)
        if vc is None or len(vc["starts"]) == 0 or n_frames == 0:
            return out

        starts = vc["starts"]
        ends = vc["ends"]
        idx = np.searchsorted(starts, frame_times, side="right") - 1
        safe_idx = np.maximum(idx, 0)
        valid = (frame_times >= 0.0) & (idx >= 0) & (frame_times < ends[safe_idx])
        if min_eval_ms is not None:
            valid &= frame_times >= min_eval_ms
        if max_eval_ms is not None:
            valid &= frame_times <= max_eval_ms

        if valid.any():
            hit = safe_idx[valid]
            out["root"][valid] = vc["roots"][hit]
            out["quality"][valid] = vc["quals"][hit]
            out["ext"][valid] = vc["exts"][hit]
            out["type_id"][valid] = vc["types"][hit]
            out["skel_id"][valid] = vc["skels"][hit]
            out["crf_state"][valid] = vc["crfs"][hit]
            out["bass"][valid] = vc["basses"][hit]
            out["true_bass"][valid] = vc["true_basses"][hit]
            out["local_key"][valid] = vc["keys"][hit]
            out["shell_chroma"][valid] = vc["shell_c"][hit]
            out["full_chroma"][valid] = vc["full_c"][hit]

        return out

    def _lookup_chord(self, song_id: str, time_ms: float) -> tuple[int, int, int, int]:
        fc = self._fast_chords.get(song_id)
        if fc is None or len(fc[0]) == 0:
            return -100, -100, -100, -100

        starts, ends, roots, quals, exts, basses = fc
        idx = int(np.searchsorted(starts, time_ms, side="right")) - 1
        if idx < 0 or time_ms >= ends[idx]:
            return -100, -100, -100, -100

        return int(roots[idx]), int(quals[idx]), int(exts[idx]), int(basses[idx])

    def _lookup_chord_frames(
        self,
        song_id: str,
        crop_start_ms: float,
        n_frames: int,
        min_eval_ms: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        fc = self._fast_chords.get(song_id)
        root_labels = np.full(n_frames, -100, dtype=np.int64)
        quality_labels = np.full(n_frames, -100, dtype=np.int64)
        ext_labels = np.full(n_frames, -100, dtype=np.int64)
        bass_labels = np.full(n_frames, -100, dtype=np.int64)

        if fc is None or len(fc[0]) == 0 or n_frames <= 0:
            return root_labels, quality_labels, ext_labels, bass_labels

        starts, ends, roots, quals, exts, basses = fc
        frame_times = mert_frame_times_ms(crop_start_ms, n_frames, self.sr)
        idx = np.searchsorted(starts, frame_times, side="right") - 1
        safe_idx = np.maximum(idx, 0)
        valid = (idx >= 0) & (frame_times < ends[safe_idx])
        if min_eval_ms is not None:
            valid &= frame_times >= min_eval_ms

        if valid.any():
            hit = safe_idx[valid]
            root_labels[valid] = roots[hit]
            quality_labels[valid] = quals[hit]
            ext_labels[valid] = exts[hit]
            bass_labels[valid] = basses[hit]

        return root_labels, quality_labels, ext_labels, bass_labels


class CachedSequenceDataset(Dataset):
    """Phase 3 Full-Song Sequence Dataset operating directly on cached 256-dim MERT+LoRA embeddings."""

    def __init__(self, base_ds: ChordDataset, cache_split_dir: Path, max_seq_s: int = 60):
        self.base = base_ds
        self.cache_dir = cache_split_dir
        self.chunk_frames = mert_output_length(base_ds.chunk_samples)
        self.max_chunks = max(1, max_seq_s // base_ds.cfg.model.chunk_s)
        self.max_frames = self.max_chunks * self.chunk_frames
        self.songs = [
            sid for sid in base_ds._valid_songs
            if (self.cache_dir / f"{sid}.npz").exists()
        ]

    def __len__(self) -> int:
        return len(self.songs)

    def __getitem__(self, idx: int) -> dict:
        sid = self.songs[idx]
        data = np.load(self.cache_dir / f"{sid}.npz")
        emb = data["emb"].astype(np.float32)          # (T_song, 256)
        rms_db = data["rms_db"].astype(np.float32)    # (T_song,)
        cqt_bins = int(getattr(self.base.cfg.model, "cqt_bins", 60))
        if "cqt" in data:
            cqt = data["cqt"].astype(np.float32)      # (T_song, 60)
        else:
            cqt = np.zeros((len(emb), cqt_bins), dtype=np.float32)
        t_song, c_dim = emb.shape
        n_song_chunks = max(1, t_song // self.chunk_frames)
        frame_times = mert_chunked_frame_times_ms(0.0, n_song_chunks, self.base.chunk_samples, self.base.sr)[:t_song]

        if t_song >= self.max_frames:
            if self.base.split == "train":
                max_c = n_song_chunks - self.max_chunks
                c0 = int(np.random.randint(0, max(1, max_c + 1)))
                f0 = c0 * self.chunk_frames
            else:
                f0 = 0
            emb_win = emb[f0: f0 + self.max_frames]
            cqt_win = cqt[f0: f0 + self.max_frames].copy()
            rms_win = rms_db[f0: f0 + self.max_frames]
            ft_win = frame_times[f0: f0 + self.max_frames]
            valid_len = self.max_frames
        else:
            emb_win = np.zeros((self.max_frames, c_dim), dtype=np.float32)
            cqt_win = np.zeros((self.max_frames, cqt_bins), dtype=np.float32)
            rms_win = np.full(self.max_frames, -75.0, dtype=np.float32)
            emb_win[:t_song] = emb
            cqt_win[: min(t_song, len(cqt))] = cqt[:t_song]
            rms_win[:t_song] = rms_db
            ft_win = np.full(self.max_frames, -1.0, dtype=np.float64)
            ft_win[:t_song] = frame_times
            valid_len = t_song

        # Phase 3 Sub-Bass Register Dropout on CQT Octaves 1-2 (simulates bassless intro -> band in 60s context)
        if self.base.split == "train" and getattr(self.base.cfg.model, "bass_first", True) and cqt_bins >= 36:
            if np.random.random() < 0.25 and valid_len > 150:
                oct1_scale = float(np.random.uniform(0.02, 0.12))
                oct2_scale = float(np.random.uniform(0.08, 0.28))
                drop_end = valid_len if np.random.random() < 0.5 else int(valid_len * np.random.uniform(0.25, 0.55))
                if cqt_bins == 96:
                    # v6: 24 bins/oct in low 3 octaves (C1..B1 = 0..23, C2..B2 = 24..47)
                    cqt_win[:drop_end, :24] *= oct1_scale
                    cqt_win[:drop_end, 24:48] *= oct2_scale
                else:
                    cqt_win[:drop_end, :12] *= oct1_scale
                    cqt_win[:drop_end, 12:24] *= oct2_scale
            spec_p = float(getattr(self.base.cfg.augment, "cqt_spec_aug_prob", 0.0))
            if self.base.is_v6 and spec_p > 0.0 and np.random.random() < spec_p:
                cqt_win[:valid_len] = apply_cqt_spec_augment(cqt_win[:valid_len])

        audio_end_ms = self.base._audio_lengths[sid] / float(self.base.sr) * 1000.0
        if self.base.is_v6:
            labels = self.base._lookup_v6_frames(
                sid,
                ft_win,
                max_eval_ms=(audio_end_ms if self.base.split != "train" else None),
            )
            sil_thresh = float(self.base.cfg.model.silence_db_threshold)
            if self.base.split == "train":
                silent = (np.arange(self.max_frames) < valid_len) & (rms_win < sil_thresh)
            else:
                silent = (
                    (np.arange(self.max_frames) < valid_len)
                    & (ft_win <= audio_end_ms)
                    & (rms_win < (sil_thresh - 3.0))
                    & (labels["root"] == -100)
                )
            if silent.any():
                labels["root"][silent] = ROOT_SILENCE_ID
                labels["true_bass"][silent] = ROOT_SILENCE_ID
                labels["skel14"][silent] = -100
                labels["tension_bits"][silent] = 0.0
                labels["tension_valid"][silent] = False
                labels["inversion"][silent] = -100
                labels["crf_state"][silent] = 0
                labels["chord_dur_ms"][silent] = 0.0
            bnd_chord, bnd_bass = self.base._compute_gaussian_boundaries(
                labels["root"], labels["skel14"], labels["true_bass"]
            )
            return {
                "emb": torch.from_numpy(emb_win),
                "cqt": torch.from_numpy(cqt_win),
                "frame_rms_db": torch.from_numpy(rms_win),
                "root": torch.from_numpy(labels["root"]),
                "quality": torch.from_numpy(labels["quality"]),
                "ext": torch.from_numpy(labels["ext"]),
                "skel14": torch.from_numpy(labels["skel14"]),
                "tension_bits": torch.from_numpy(labels["tension_bits"]),
                "tension_valid": torch.from_numpy(labels["tension_valid"]),
                "inversion": torch.from_numpy(labels["inversion"]),
                "crf_state": torch.from_numpy(labels["crf_state"]),
                "bass": torch.from_numpy(labels["bass"]),
                "true_bass": torch.from_numpy(labels["true_bass"]),
                "local_key": torch.from_numpy(labels["local_key"]),
                "shell_chroma": torch.from_numpy(labels["shell_chroma"]),
                "full_chroma": torch.from_numpy(labels["full_chroma"]),
                "boundary_chord": torch.from_numpy(bnd_chord),
                "boundary_bass": torch.from_numpy(bnd_bass),
                "chord_dur_ms": torch.from_numpy(labels["chord_dur_ms"]),
                "song_id": sid,
            }

        labels = self.base._lookup_v5_frames(
            sid,
            ft_win,
            max_eval_ms=(audio_end_ms if self.base.split != "train" else None),
        )

        sil_thresh = float(self.base.cfg.model.silence_db_threshold)
        if self.base.split == "train":
            silent = (np.arange(self.max_frames) < valid_len) & (rms_win < sil_thresh)
        else:
            silent = (np.arange(self.max_frames) < valid_len) & (ft_win <= audio_end_ms) & (rms_win < (sil_thresh - 3.0)) & (labels["root"] == -100)

        if silent.any():
            labels["root"][silent] = ROOT_SILENCE_ID
            labels["true_bass"][silent] = ROOT_SILENCE_ID
            labels["crf_state"][silent] = 0

        return {
            "emb": torch.from_numpy(emb_win),
            "cqt": torch.from_numpy(cqt_win),
            "frame_rms_db": torch.from_numpy(rms_win),
            "root": torch.from_numpy(labels["root"]),
            "quality": torch.from_numpy(labels["quality"]),
            "ext": torch.from_numpy(labels["ext"]),
            "type_id": torch.from_numpy(labels["type_id"]),
            "skel_id": torch.from_numpy(labels["skel_id"]),
            "crf_state": torch.from_numpy(labels["crf_state"]),
            "bass": torch.from_numpy(labels["bass"]),
            "true_bass": torch.from_numpy(labels["true_bass"]),
            "local_key": torch.from_numpy(labels["local_key"]),
            "shell_chroma": torch.from_numpy(labels["shell_chroma"]),
            "full_chroma": torch.from_numpy(labels["full_chroma"]),
            "song_id": sid,
        }
