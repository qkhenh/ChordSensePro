from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.dataset import ChordDataset
from src.model import mert_output_length


@pytest.fixture
def dummy_data(tmp_path):
    sr = 24000
    crop_s = 5
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
    assert "song_short" not in ds._valid_songs


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


def test_val_deterministic_and_tiled(dummy_data):
    data_dir, sr, crop_s = dummy_data
    cfg = Config(data_dir=str(data_dir), crop_s=crop_s, sample_rate=sr)
    cfg.model.frame_level = True
    ds = ChordDataset(cfg, split="val")
    # Each 12s song gets 3 tiles: [0..5s], [5..10s], and tail [7..12s] with prefix [7..10s) masked
    assert len(ds) == 5 * 3
    item1 = ds[0]
    item2 = ds[0]
    assert (item1["wav"] == item2["wav"]).all()
    assert (item1["root"] == item2["root"]).all()
    assert (item1["bass"] == item2["bass"]).all()

    tail_item = ds[2]
    # Prefix frames before 10000ms must be masked (-100), tail frames >= 10000ms must be valid
    assert (tail_item["root"][:100] == -100).all()
    assert (tail_item["root"][-50:] >= 0).all()


def test_multi_crop_long_song_and_padding_bounds(tmp_path):
    sr = 24000
    crop_s = 5
    audio_dir = tmp_path / "train" / "audio"
    audio_dir.mkdir(parents=True)

    # 70s audio with 3s unlabelled padding on both ends -> labelled [3000, 63000] (60s -> 4 crops)
    audio = np.random.randint(-20000, 20000, size=sr * 70, dtype=np.int16)
    np.save(audio_dir / "long_jazz.npy", audio)
    df = pd.DataFrame([{
        "song_id": "long_jazz",
        "start_ms": 3000,
        "end_ms": 63000,
        "root_id": 2,
        "quality_id": 1,
        "ext_id": 4,
        "bass_id": 9,
    }])
    df.to_parquet(tmp_path / "train" / "chords.parquet")

    cfg = Config(data_dir=str(tmp_path), crop_s=crop_s, sample_rate=sr)
    cfg.model.frame_level = True
    ds = ChordDataset(cfg, split="train")
    assert len(ds) == 4
    lo, hi = ds._crop_bounds["long_jazz"]
    assert lo == 3 * sr
    assert hi == (63 - crop_s) * sr
    item = ds[0]
    assert (item["root"] == 2).all()
    assert (item["bass"] == 9).all()


def test_lookup_no_chord_gap(dummy_data):
    data_dir, sr, crop_s = dummy_data
    cfg = Config(data_dir=str(data_dir), crop_s=crop_s, sample_rate=sr)
    ds = ChordDataset(cfg, split="train")
    result = ds._lookup_chord("nonexistent_song", 1000.0)
    assert result == (-100, -100, -100, -100)


def test_frame_level_dataset(dummy_data):
    data_dir, sr, crop_s = dummy_data
    cfg = Config(data_dir=str(data_dir), crop_s=crop_s, sample_rate=sr)
    cfg.model.frame_level = True
    ds = ChordDataset(cfg, split="train")
    item = ds[0]
    expected_frames = mert_output_length(sr * crop_s)
    assert expected_frames == 374
    assert item["root"].shape == (expected_frames,)
    assert item["quality"].shape == (expected_frames,)
    assert item["ext"].shape == (expected_frames,)
    assert item["bass"].shape == (expected_frames,)

    for i in range(expected_frames):
        r = item["root"][i].item()
        q = item["quality"][i].item()
        e = item["ext"][i].item()
        b = item["bass"][i].item()
        assert r == -100 or (0 <= r < 12)
        assert q == -100 or (0 <= q < 6)
        assert e == -100 or (0 <= e < 15)
        assert b == -100 or (0 <= b < 13)


def test_frame_level_lookup_empty(dummy_data):
    data_dir, sr, crop_s = dummy_data
    cfg = Config(data_dir=str(data_dir), crop_s=crop_s, sample_rate=sr)
    cfg.model.frame_level = True
    ds = ChordDataset(cfg, split="train")
    r, q, e, b = ds._lookup_chord_frames("nonexistent_song", 0.0, 374)
    assert (r == -100).all()
    assert (q == -100).all()
    assert (e == -100).all()
    assert (b == -100).all()
