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
    INVERSION_NAMES_5D,
    SKELETON_NAMES_V6,
    TENSION_NAMES_7D,
    parse_v6_chord_targets,
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
    if not src_audio_dir.exists():
        return 0
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


def annotate_v4_dataframe(
    df: pd.DataFrame,
    source_domain: str = "demucs",
    is_synthetic_rare: bool = False,
) -> pd.DataFrame:
    """Add v4 / Model v6 decomposed target columns to a v3-compatible chord DataFrame."""
    out = df.copy()
    has_bass = "bass_id" in out.columns
    has_raw = "label_raw" in out.columns

    skel14_ids: list[int] = []
    tension_masks: list[int] = []
    tension_valids: list[bool] = []
    bass_ids: list[int] = []
    inv_ids: list[int] = []
    missing_thirds: list[bool] = []

    raw_col = out["label_raw"] if has_raw else [None] * len(out)
    bass_col = out["bass_id"] if has_bass else [BASS_NONE_ID] * len(out)

    for raw, rid, qid, eid, bid in zip(
        raw_col,
        out["root_id"],
        out["quality_id"],
        out["ext_id"],
        bass_col,
    ):
        t = parse_v6_chord_targets(
            raw if pd.notna(raw) else None,
            int(rid) if pd.notna(rid) else None,
            int(qid) if pd.notna(qid) else None,
            int(eid) if pd.notna(eid) else None,
            int(bid) if pd.notna(bid) else BASS_NONE_ID,
        )
        skel14_ids.append(t.skel14_id)
        tension_masks.append(t.tension_mask)
        tension_valids.append(t.tension_valid)
        bass_ids.append(t.bass_id)
        inv_ids.append(t.inversion_id)
        missing_thirds.append(t.missing_third)

    out["bass_id"] = pd.array(bass_ids, dtype="int64")
    out["skel14_id"] = pd.array(skel14_ids, dtype="int64")
    out["tension_mask"] = pd.array(tension_masks, dtype="uint8")
    out["tension_valid"] = pd.array(tension_valids, dtype="bool")
    out["inversion_id"] = pd.array(inv_ids, dtype="int64")
    out["missing_third"] = pd.array(missing_thirds, dtype="bool")
    if "is_synthetic_rare" not in out.columns:
        out["is_synthetic_rare"] = pd.array([is_synthetic_rare] * len(out), dtype="bool")
    if "source_domain" not in out.columns:
        out["source_domain"] = source_domain
    return out


def _merge_optional_pack(
    split_dfs: dict[str, pd.DataFrame],
    pack_dir: Path,
    out_dir: Path,
    stats_only: bool,
) -> dict[str, int]:
    """Merge optional POP909 piano pack & rare synthetic pack into v4 without touching `test`."""
    extra_counts: dict[str, int] = {}
    for split_name in ("train", "val", "test_piano_pop"):
        pq_path = pack_dir / split_name / "chords.parquet"
        if not pq_path.exists():
            continue
        pack_df = pd.read_parquet(pq_path)
        if "skel14_id" not in pack_df.columns:
            pack_df = annotate_v4_dataframe(
                pack_df,
                source_domain=str(pack_df["source_domain"].iloc[0]) if "source_domain" in pack_df.columns and len(pack_df) else "piano_pop",
                is_synthetic_rare=False,
            )
        extra_counts[split_name] = int(pack_df["song_id"].nunique())
        if split_name in split_dfs:
            split_dfs[split_name] = pd.concat([split_dfs[split_name], pack_df], ignore_index=True)
        else:
            split_dfs[split_name] = pack_df
        if not stats_only:
            n_audio = link_or_copy_audio(pack_dir / split_name / "audio", out_dir / split_name / "audio")
            print(f"[pack:{split_name}] merged {len(pack_df):,} chords ({extra_counts[split_name]} songs), linked {n_audio:,} audio files")
    return extra_counts


def export_from_v3(
    v3_dir: Path,
    out_dir: Path,
    piano_pack_dir: Path | None = None,
    stats_only: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    split_dfs: dict[str, pd.DataFrame] = {}

    for split in ("train", "val", "test"):
        pq_path = v3_dir / split / "chords.parquet"
        if not pq_path.exists():
            raise FileNotFoundError(f"Missing {pq_path}")
        df = pd.read_parquet(pq_path)
        split_dfs[split] = annotate_v4_dataframe(df, source_domain="demucs", is_synthetic_rare=False)

    if piano_pack_dir is not None and piano_pack_dir.exists():
        _merge_optional_pack(split_dfs, piano_pack_dir, out_dir, stats_only=stats_only)

    counts: dict[str, int] = {}
    skel_counter: Counter[int] = Counter()
    inv_counter: Counter[int] = Counter()
    bass_counter: Counter[int] = Counter()
    tension_counts = [0] * len(TENSION_NAMES_7D)
    total_chords = 0
    rescued_sixth = 0
    missing_third_cnt = 0
    short_500 = 0
    short_800 = 0

    for split, df in split_dfs.items():
        counts[split] = int(df["song_id"].nunique())
        total_chords += len(df)
        skel_counter.update(int(x) for x in df["skel14_id"])
        inv_counter.update(int(x) for x in df["inversion_id"])
        bass_counter.update(int(x) for x in df["bass_id"])
        missing_third_cnt += int(df["missing_third"].sum())
        rescued_sixth += int(((df["skel14_id"] == 1) | (df["skel14_id"] == 6)).sum())
        if "start_ms" in df.columns and "end_ms" in df.columns:
            durs = df["end_ms"] - df["start_ms"]
            short_500 += int((durs <= 500).sum())
            short_800 += int((durs <= 800).sum())
        t_masks = df.loc[df["tension_valid"], "tension_mask"].to_numpy()
        for b_idx in range(len(TENSION_NAMES_7D)):
            tension_counts[b_idx] += int(((t_masks & (1 << b_idx)) != 0).sum())

    slash_chords = sum(n for bid, n in bass_counter.items() if bid != BASS_NONE_ID)
    print(f"Total chords: {total_chords:,} | Songs: {counts}")
    print(
        f"Rescued 6th-family (MAJ6+MIN6): {rescued_sixth:,} | Missing-3rd power chords: {missing_third_cnt:,} | "
        f"Slash chords: {slash_chords:,} ({100.0 * slash_chords / max(total_chords, 1):.2f}%)"
    )
    print(
        f"Passing chords: <=500ms = {short_500:,} ({100.0 * short_500 / max(total_chords, 1):.1f}%), "
        f"<=800ms = {short_800:,} ({100.0 * short_800 / max(total_chords, 1):.1f}%)"
    )

    print("\n14-Skeleton distribution:")
    for i, name in enumerate(SKELETON_NAMES_V6):
        n = skel_counter.get(i, 0)
        print(f"  {i:>2} {name:<12} {n:>8,} ({100.0 * n / max(total_chords, 1):5.2f}%)")

    print("\n7-Tension bit counts:")
    for i, name in enumerate(TENSION_NAMES_7D):
        print(f"  bit {i} {name:<8} {tension_counts[i]:>8,}")

    print("\n5-Inversion distribution:")
    for i, name in enumerate(INVERSION_NAMES_5D):
        n = inv_counter.get(i, 0)
        print(f"  {i:>2} {name:<12} {n:>8,} ({100.0 * n / max(total_chords, 1):5.2f}%)")

    if stats_only:
        return

    for split, df in split_dfs.items():
        split_out = out_dir / split
        split_out.mkdir(parents=True, exist_ok=True)
        df.to_parquet(split_out / "chords.parquet", index=False)
        if split in ("train", "val", "test"):
            n_audio = link_or_copy_audio(v3_dir / split / "audio", split_out / "audio")
            print(f"[{split}] wrote {len(df):,} chords, linked {n_audio:,} base v3 audio files")

    config = {
        "sample_rate": 24000,
        "crop_s": 5,
        "dtype": "int16",
        "mono": True,
        "variable_length": True,
        "label_status_included": ["ok", "partial"],
        "pipeline_version": 4,
    }
    (out_dir / "config.yaml").write_text(
        "\n".join(f"{k}: {json.dumps(v)}" for k, v in config.items()) + "\n",
        encoding="utf-8",
    )

    classes_payload = {
        "root": [r.value for r in RootNote],
        "quality": [q.value for q in ChordQuality],
        "extension": [e.value for e in ChordExtension],
        "bass": BASS_NAMES,
        "valid_combo_mask": valid_combo_mask(),
        "skeletons_v6": list(SKELETON_NAMES_V6),
        "tensions_7d": list(TENSION_NAMES_7D),
        "inversions_5d": list(INVERSION_NAMES_5D),
        "counts": counts,
        "bass_distribution": {str(i): int(bass_counter.get(i, 0)) for i in range(len(BASS_NAMES))},
        "skel14_distribution": {SKELETON_NAMES_V6[i]: int(skel_counter.get(i, 0)) for i in range(len(SKELETON_NAMES_V6))},
        "tension_distribution": {TENSION_NAMES_7D[i]: int(tension_counts[i]) for i in range(len(TENSION_NAMES_7D))},
        "inversion_distribution": {INVERSION_NAMES_5D[i]: int(inv_counter.get(i, 0)) for i in range(len(INVERSION_NAMES_5D))},
    }
    (out_dir / "classes.json").write_text(json.dumps(classes_payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nExported v4 dataset ({counts}) -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export Dataset v4 (14 Skeletons, 7 Tensions, 5 Inversions)")
    ap.add_argument("--v3-dir", default="DE/data/export/v3", help="Source v3 export directory")
    ap.add_argument("--out", default="DE/data/export/v4", help="Target v4 export directory")
    ap.add_argument(
        "--piano-pack-dir",
        default=None,
        help="Optional POP909 piano & rare-chord pack directory from build_v4_piano_and_rare_pack.py",
    )
    ap.add_argument("--stats", action="store_true", help="Print v4 label statistics only without writing files")
    args = ap.parse_args()

    export_from_v3(
        Path(args.v3_dir),
        Path(args.out),
        piano_pack_dir=Path(args.piano_pack_dir) if args.piano_pack_dir else None,
        stats_only=args.stats,
    )


if __name__ == "__main__":
    main()
