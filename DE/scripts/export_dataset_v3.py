from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/opt/airflow")

import pandas as pd

from src.data_processing.application.mappers.harte_mapper import (
    BASS_NONE_ID,
    parse_bass_id,
)
from src.shared.domain.value_objects import (
    ChordExtension,
    ChordQuality,
    RootNote,
    valid_combo_mask,
)

BASS_NAMES = [r.value for r in RootNote] + ["none"]


def link_or_copy_audio(src_audio_dir: Path, dst_audio_dir: Path) -> int:
    dst_audio_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for src_file in src_audio_dir.glob("*.npy"):
        dst_file = dst_audio_dir / src_file.name
        if dst_file.exists():
            count += 1
            continue
        try:
            os.link(src_file, dst_file)
        except OSError:
            shutil.copy2(src_file, dst_file)
        count += 1
    return count


def export_from_v2(v2_dir: Path, out_dir: Path, stats_only: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    bass_counter: Counter[int] = Counter()
    split_dfs: dict[str, pd.DataFrame] = {}

    for split in ("train", "val", "test"):
        pq_path = v2_dir / split / "chords.parquet"
        if not pq_path.exists():
            raise FileNotFoundError(f"Missing {pq_path}")

        df = pd.read_parquet(pq_path)
        bass_ids = [
            parse_bass_id(
                raw if pd.notna(raw) else None,
                int(rid) if pd.notna(rid) else None,
            )
            for raw, rid in zip(df["label_raw"], df["root_id"])
        ]
        df["bass_id"] = pd.array(bass_ids, dtype="int64")
        split_dfs[split] = df
        counts[split] = int(df["song_id"].nunique())
        bass_counter.update(bass_ids)

    total_chords = sum(len(df) for df in split_dfs.values())
    slash_chords = sum(n for bid, n in bass_counter.items() if bid != BASS_NONE_ID)
    print(f"Total chords: {total_chords:,} | Slash chords (bass != root): {slash_chords:,} ({100.0 * slash_chords / max(total_chords, 1):.2f}%)")
    print("\nBass head distribution (13 classes):")
    for i, name in enumerate(BASS_NAMES):
        n = bass_counter.get(i, 0)
        pct = 100.0 * n / max(total_chords, 1)
        print(f"  {i:>2} {name:<6} {n:>8,} ({pct:5.2f}%)")

    if stats_only:
        return

    for split, df in split_dfs.items():
        split_out = out_dir / split
        split_out.mkdir(parents=True, exist_ok=True)
        df.to_parquet(split_out / "chords.parquet", index=False)
        n_audio = link_or_copy_audio(v2_dir / split / "audio", split_out / "audio")
        print(f"[{split}] wrote {len(df):,} chords, linked {n_audio:,} audio files")

    config = {
        "sample_rate": 24000,
        "crop_s": 5,
        "dtype": "int16",
        "mono": True,
        "variable_length": True,
        "label_status_included": ["ok", "partial"],
        "pipeline_version": 3,
    }
    (out_dir / "config.yaml").write_text(
        "\n".join(f"{k}: {json.dumps(v)}" for k, v in config.items()) + "\n",
        encoding="utf-8",
    )

    bass_dist = {str(i): int(bass_counter.get(i, 0)) for i in range(len(BASS_NAMES))}
    (out_dir / "classes.json").write_text(
        json.dumps(
            {
                "root": [r.value for r in RootNote],
                "quality": [q.value for q in ChordQuality],
                "extension": [e.value for e in ChordExtension],
                "bass": BASS_NAMES,
                "valid_combo_mask": valid_combo_mask(),
                "counts": counts,
                "bass_distribution": bass_dist,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nExported v3 dataset ({counts}) -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export dataset v3 with bass_id (13 classes)")
    ap.add_argument("--v2-dir", default="DE/data/export/v2", help="Source v2 export directory")
    ap.add_argument("--out", default="DE/data/export/v3", help="Target v3 export directory")
    ap.add_argument("--stats", action="store_true", help="Print bass distribution only")
    args = ap.parse_args()

    export_from_v2(Path(args.v2_dir), Path(args.out), stats_only=args.stats)


if __name__ == "__main__":
    main()

