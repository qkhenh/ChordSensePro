from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.augment import pitch_shift_bass, pitch_shift_root
from src.config import Config
from src.model import mert_frame_times_ms, mert_output_length


class ChordDataset(Dataset):

    def __init__(self, cfg: Config, split: str = "train", augment_fn=None):
        self.cfg = cfg
        self.split = split
        self.augment_fn = augment_fn if split == "train" else None
        self.crop_samples = cfg.crop_samples
        self.sr = cfg.sample_rate

        split_dir = Path(cfg.data_dir) / split
        chords_df = pd.read_parquet(split_dir / "chords.parquet")

        self.audio_dir = split_dir / "audio"
        self.song_ids = sorted(chords_df["song_id"].unique())
        self.song_to_idx = {sid: i for i, sid in enumerate(self.song_ids)}

        self._chords_by_song: dict[str, pd.DataFrame] = {}
        self._fast_chords: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
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

        self._audio_lengths: dict[str, int] = {}
        for sid in self.song_ids:
            npy_path = self.audio_dir / f"{sid}.npy"
            header = np.lib.format.open_memmap(str(npy_path), mode="r")
            self._audio_lengths[sid] = len(header)
            del header

        self._valid_songs = [
            sid for sid in self.song_ids
            if self._audio_lengths[sid] >= self.crop_samples
        ]

        # Precompute labelled crop bounds [lo_sample, hi_sample] per song & sample index list
        self._crop_bounds: dict[str, tuple[int, int]] = {}
        # Each entry: (song_id, fixed_crop_start_or_None, min_eval_ms_or_None)
        self._samples: list[tuple[str, int | None, float | None]] = []

        for sid in self._valid_songs:
            audio_len = self._audio_lengths[sid]
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
                # Deterministic non-overlapping tiles across [lo, hi]
                tiles: list[tuple[int, float | None]] = []
                pos = lo
                prev_end_ms: float | None = None
                while pos <= hi:
                    tiles.append((pos, prev_end_ms))
                    prev_end_ms = (pos + self.crop_samples) / float(self.sr) * 1000.0
                    if pos == hi:
                        break
                    if pos + self.crop_samples > hi:
                        # Tail window anchored at hi; mask prefix already covered by previous tile
                        tiles.append((hi, prev_end_ms))
                        break
                    pos += self.crop_samples

                # Cap validation tiles on very long tracks to keep per-epoch val fast;
                # `test` keeps 100% full-song coverage.
                max_val_tiles = 6
                if self.split == "val" and len(tiles) > max_val_tiles:
                    idxs = np.linspace(0, len(tiles) - 1, max_val_tiles, dtype=int)
                    for ti in idxs:
                        # Evenly spaced non-adjacent tiles don't overlap -> no prefix mask needed
                        self._samples.append((sid, tiles[int(ti)][0], None))
                else:
                    for t_pos, t_mask in tiles:
                        self._samples.append((sid, t_pos, t_mask))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
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
