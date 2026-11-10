from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.dataset import ChordDataset


@pytest.fixture
def dummy_data(tmp_path):
    sr = 24000
    crop_s = 5
    crop_samples = sr * crop_s
    song_len = sr * 12

    for split in ["train", "val"]:
        audio_dir = tmp_path / split / "audio"
        audio_dir.mkdir(parents=True)

        records = []
        for i in range(5):
            sid = f"song_{i:03d}"
            audio = np.random.randint(-30000, 30000, size=song_len, dtype=np.int16)
            np.save(audio_dir / f"{sid}.npy", audio)

            n_chords = 6
            dur_ms = song_len / sr * 1000
            step_ms = dur_ms / n_chords
            for j in range(n_chords):
                records.append({
                    "song_id": sid,
                    "start_ms": j * step_ms,
                    "end_ms": (j + 1) * step_ms,
                    "root_id": j % 12,
                    "quality_id": j % 6,
                    "ext_id": j % 15,
                    "bass_id": 12 if j % 2 == 0 else (j + 4) % 12,
                })

        sid_short = "song_short"
        short_audio = np.random.randint(-30000, 30000, size=sr, dtype=np.int16)
        np.save(audio_dir / f"{sid_short}.npy", short_audio)
        records.append({
            "song_id": sid_short,
            "start_ms": 0,
            "end_ms": 1000,
            "root_id": 0,
            "quality_id": 0,
            "ext_id": 0,
            "bass_id": 12,
        })

        df = pd.DataFrame(records)
        df.to_parquet(tmp_path / split / "chords.parquet")

    return tmp_path, sr, crop_s


def test_dataset_length(dummy_data):
    data_dir, sr, crop_s = dummy_data
    cfg = Config(data_dir=str(data_dir), crop_s=crop_s, sample_rate=sr)
    ds = ChordDataset(cfg, split="train")
    assert len(ds) == 5


def test_short_songs_excluded(dummy_data):
    data_dir, sr, crop_s = dummy_data
    cfg = Config(data_dir=str(data_dir), crop_s=crop_s, sample_rate=sr)
    ds = ChordDataset(cfg, split="train")
    song_ids = [ds._valid_songs[i] for i in range(len(ds))]
    assert "song_short" not in song_ids


def test_crop_shape(dummy_data):
    data_dir, sr, crop_s = dummy_data
    cfg = Config(data_dir=str(data_dir), crop_s=crop_s, sample_rate=sr)
    ds = ChordDataset(cfg, split="train")
    item = ds[0]
    assert item["wav"].shape == (sr * crop_s,)
    assert item["wav"].dtype == __import__("torch").float32


def test_labels_valid_range(dummy_data):
    data_dir, sr, crop_s = dummy_data
    cfg = Config(data_dir=str(data_dir), crop_s=crop_s, sample_rate=sr)
    ds = ChordDataset(cfg, split="train")
    for i in range(len(ds)):
        item = ds[i]
        root = item["root"].item()
        quality = item["quality"].item()
        ext = item["ext"].item()
        bass = item["bass"].item()
        assert root == -100 or (0 <= root < 12)
        assert quality == -100 or (0 <= quality < 6)
        assert ext == -100 or (0 <= ext < 15)
        assert bass == -100 or (0 <= bass < 13)


def test_val_deterministic(dummy_data):
    data_dir, sr, crop_s = dummy_data
    cfg = Config(data_dir=str(data_dir), crop_s=crop_s, sample_rate=sr)
    ds = ChordDataset(cfg, split="val")
    item1 = ds[0]
    item2 = ds[0]
    assert (item1["wav"] == item2["wav"]).all()
    assert item1["root"] == item2["root"]
    assert item1["bass"] == item2["bass"]


def test_lookup_no_chord_gap(dummy_data):
    data_dir, sr, crop_s = dummy_data
    cfg = Config(data_dir=str(data_dir), crop_s=crop_s, sample_rate=sr)
    ds = ChordDataset(cfg, split="train")
    result = ds._lookup_chord("nonexistent_song", 1000.0)
    assert result == (-100, -100, -100, -100)

