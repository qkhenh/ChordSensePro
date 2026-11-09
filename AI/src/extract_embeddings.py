from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from transformers import AutoModel

from src.config import Config, load_config

log = logging.getLogger(__name__)


def extract_split(
    model: AutoModel,
    cfg: Config,
    split: str,
    device: torch.device,
    out_dir: Path,
    n_crops: int = 3,
) -> None:
    split_dir = Path(cfg.data_dir) / split
    chords_df = pd.read_parquet(split_dir / "chords.parquet")
    audio_dir = split_dir / "audio"
    song_ids = sorted(chords_df["song_id"].unique())

    crop_len = cfg.crop_samples
    embeddings_all = []
    labels_all = []

    chords_by_song = {}
    for sid, group in chords_df.groupby("song_id"):
        chords_by_song[sid] = group.sort_values("start_ms").reset_index(drop=True)

    for i, sid in enumerate(song_ids):
        npy_path = audio_dir / f"{sid}.npy"
        if not npy_path.exists():
            continue
        audio = np.load(npy_path)
        audio_len = len(audio)

        if audio_len < crop_len:
            continue

        if split == "train":
            max_start = audio_len - crop_len
            starts = [int(max_start * j / max(n_crops - 1, 1)) for j in range(n_crops)]
        else:
            mid = (audio_len - crop_len) // 2
            starts = [mid]

        chords = chords_by_song.get(sid)

        for s in starts:
            crop = audio[s: s + crop_len].astype(np.float32) / 32768.0
            std = crop.std()
            if std > 1e-6:
                crop = (crop - crop.mean()) / std

            wav_t = torch.from_numpy(crop).unsqueeze(0).to(device)

            with torch.no_grad(), autocast("cuda"):
                out = model(wav_t, output_hidden_states=True, return_dict=True)

            stacked = torch.stack(out.hidden_states)
            pooled = stacked.mean(dim=2).squeeze(1)

            embeddings_all.append(pooled.cpu())

            mid_ms = s / cfg.sample_rate * 1000 + cfg.crop_s * 500
            root_id, qual_id, ext_id = _lookup(chords, mid_ms)
            labels_all.append((sid, s, root_id, qual_id, ext_id))

        if (i + 1) % 100 == 0:
            log.info(f"  [{split}] {i+1}/{len(song_ids)} songs processed")

    emb_tensor = torch.stack(embeddings_all)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(emb_tensor, out_dir / f"{split}_embeddings.pt")
    torch.save(labels_all, out_dir / f"{split}_labels.pt")
    log.info(f"[{split}] Saved {emb_tensor.shape[0]} embeddings, shape={emb_tensor.shape}")


def _lookup(chords: pd.DataFrame | None, time_ms: float) -> tuple[int, int, int]:
    if chords is None or chords.empty:
        return -100, -100, -100
    mask = (chords["start_ms"] <= time_ms) & (chords["end_ms"] > time_ms)
    matches = chords.loc[mask]
    if matches.empty:
        return -100, -100, -100
    row = matches.iloc[0]
    root_id = int(row["root_id"]) if pd.notna(row["root_id"]) else -100
    qual_id = int(row["quality_id"]) if pd.notna(row["quality_id"]) else -100
    ext_id = int(row["ext_id"]) if pd.notna(row["ext_id"]) else -100
    return root_id, qual_id, ext_id


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out_dir", default="results/embeddings")
    parser.add_argument("--n_crops", type=int, default=3)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info(f"Loading MERT model: {cfg.model.name}")
    model = AutoModel.from_pretrained(cfg.model.name, trust_remote_code=True)
    model = model.to(device).eval()

    out_dir = Path(args.out_dir)
    for split in ["train", "val", "test"]:
        log.info(f"Extracting {split}...")
        extract_split(model, cfg, split, device, out_dir, n_crops=args.n_crops)

    log.info("Done.")


if __name__ == "__main__":
    main()

