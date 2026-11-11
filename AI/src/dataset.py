from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.augment import pitch_shift_bass, pitch_shift_root
from src.config import Config


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
        for sid, group in chords_df.groupby("song_id"):
            self._chords_by_song[sid] = group.sort_values("start_ms").reset_index(drop=True)

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

    def __len__(self) -> int:
        return len(self._valid_songs)

    def __getitem__(self, idx: int) -> dict:
        song_id = self._valid_songs[idx]
        audio_len = self._audio_lengths[song_id]

        max_start = audio_len - self.crop_samples
        if self.split == "train":
            crop_start = np.random.randint(0, max_start + 1)
        else:
            crop_start = max_start // 2

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
            n_frames = int(self.crop_samples / self.sr * 75)
            root_arr, qual_arr, ext_arr, bass_arr = self._lookup_chord_frames(song_id, crop_start_ms, n_frames)
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
        chords = self._chords_by_song.get(song_id)
        if chords is None or chords.empty:
            return -100, -100, -100, -100

        mask = (chords["start_ms"] <= time_ms) & (chords["end_ms"] > time_ms)
        matches = chords.loc[mask]

        if matches.empty:
            return -100, -100, -100, -100

        row = matches.iloc[0]

        root_id = int(row["root_id"]) if pd.notna(row["root_id"]) else -100
        quality_id = int(row["quality_id"]) if pd.notna(row["quality_id"]) else -100
        ext_id = int(row["ext_id"]) if pd.notna(row["ext_id"]) else -100
        if "bass_id" in row and pd.notna(row["bass_id"]):
            bass_id = int(row["bass_id"])
        else:
            bass_id = -100 if root_id == -100 else 12

        return root_id, quality_id, ext_id, bass_id

    def _lookup_chord_frames(
        self, song_id: str, crop_start_ms: float, n_frames: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        chords = self._chords_by_song.get(song_id)
        if chords is None or chords.empty:
            return (
                np.full(n_frames, -100, dtype=np.int64),
                np.full(n_frames, -100, dtype=np.int64),
                np.full(n_frames, -100, dtype=np.int64),
                np.full(n_frames, -100, dtype=np.int64),
            )

        frame_times = crop_start_ms + np.arange(n_frames) * (1000.0 / 75.0)
        starts = chords["start_ms"].values
        ends = chords["end_ms"].values
        root_labels = np.full(n_frames, -100, dtype=np.int64)
        quality_labels = np.full(n_frames, -100, dtype=np.int64)
        ext_labels = np.full(n_frames, -100, dtype=np.int64)
        bass_labels = np.full(n_frames, -100, dtype=np.int64)

        for ci in range(len(chords)):
            mask = (frame_times >= starts[ci]) & (frame_times < ends[ci])
            if not mask.any():
                continue
            row = chords.iloc[ci]
            r = int(row["root_id"]) if pd.notna(row["root_id"]) else -100
            q = int(row["quality_id"]) if pd.notna(row["quality_id"]) else -100
            e = int(row["ext_id"]) if pd.notna(row["ext_id"]) else -100
            if "bass_id" in row and pd.notna(row["bass_id"]):
                b = int(row["bass_id"])
            else:
                b = -100 if r == -100 else 12
            root_labels[mask] = r
            quality_labels[mask] = q
            ext_labels[mask] = e
            bass_labels[mask] = b

        return root_labels, quality_labels, ext_labels, bass_labels

