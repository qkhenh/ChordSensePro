from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

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

        crop_start_ms = crop_start / self.sr * 1000
        mid_ms = crop_start_ms + (self.cfg.crop_s * 500)
        root_id, quality_id, ext_id = self._lookup_chord(song_id, mid_ms)

        pitch_shift_n = 0
        if self.augment_fn is not None:
            crop, pitch_shift_n = self.augment_fn(crop, self.sr)

        if pitch_shift_n != 0 and root_id != -100:
            root_id = (root_id + pitch_shift_n) % 12

        std = crop.std()
        if std > 1e-6:
            crop = (crop - crop.mean()) / std

        return {
            "wav": torch.from_numpy(crop),
            "root": torch.tensor(root_id, dtype=torch.long),
            "quality": torch.tensor(quality_id, dtype=torch.long),
            "ext": torch.tensor(ext_id, dtype=torch.long),
            "song_id": song_id,
        }

    def _lookup_chord(self, song_id: str, time_ms: float) -> tuple[int, int, int]:
        chords = self._chords_by_song.get(song_id)
        if chords is None or chords.empty:
            return -100, -100, -100

        mask = (chords["start_ms"] <= time_ms) & (chords["end_ms"] > time_ms)
        matches = chords.loc[mask]

        if matches.empty:
            return -100, -100, -100

        row = matches.iloc[0]

        root_id = int(row["root_id"]) if pd.notna(row["root_id"]) else -100
        quality_id = int(row["quality_id"]) if pd.notna(row["quality_id"]) else -100
        ext_id = int(row["ext_id"]) if pd.notna(row["ext_id"]) else -100

        return root_id, quality_id, ext_id

