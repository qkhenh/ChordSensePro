from __future__ import annotations
import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "/opt/airflow")

import numpy as np

from src.shared.domain.value_objects import (
    VALID_COMBOS, ChordExtension, ChordQuality, RootNote, valid_combo_mask,
)
from src.shared.infrastructure.postgres.client import get_connection
from src.data_processing.application.pipeline_handlers.sample_writer_handler import (
    training_v2_dir,
)

SECTION_COLUMNS = [
    "song_id", "source_type", "source_url", "audio_path", "raw_audio_path",
    "duration_ms", "align_offset_ms", "align_score", "align_status", "split",
]

CHORD_COLUMNS = [
    "song_id", "start_ms", "end_ms", "label_raw", "label_status",
    "root_id", "quality_id", "ext_id",
]


def fetch_sections(statuses: tuple[str, ...]) -> list[dict]:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT {', '.join(SECTION_COLUMNS)} FROM training_sections "
            "WHERE align_status != 'rejected' ORDER BY split, song_id",
        )
        return [dict(zip(SECTION_COLUMNS, r)) for r in cur.fetchall()]


def fetch_chords(song_ids: list[str], statuses: tuple[str, ...]) -> dict[str, list[dict]]:
    if not song_ids:
        return {}
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT {', '.join(CHORD_COLUMNS)} FROM training_chords "
            "WHERE song_id = ANY(%s) AND label_status = ANY(%s) "
            "ORDER BY song_id, start_ms",
            (song_ids, list(statuses)),
        )
        by_song: dict[str, list[dict]] = defaultdict(list)
        for r in cur.fetchall():
            row = dict(zip(CHORD_COLUMNS, r))
            by_song[row["song_id"]].append(row)
        return by_song


def drop_invalid_combos(chords: list[dict]) -> list[dict]:
    qualities, exts = list(ChordQuality), list(ChordExtension)
    kept, dropped = [], []
    for c in chords:
        q, e = c["quality_id"], c["ext_id"]
        if q is not None and e is not None and (qualities[q], exts[e]) not in VALID_COMBOS:
            dropped.append(c)
        else:
            kept.append(c)
    if dropped:
        pct = 100.0 * len(dropped) / max(len(chords), 1)
        print(f"\ninvalid combos dropped: {len(dropped)} of {len(chords)} ({pct:.2f}%)")
    return kept


def print_distribution(sections: list[dict], all_chords: list[dict]) -> None:
    by_split = Counter(s["split"] for s in sections)
    print(f"\nsections per split: {dict(by_split)}")
    print(f"per source: {dict(Counter(s['source_type'] for s in sections))}")
    print(f"total chords: {len(all_chords)}")
    for head, names, key in (
        ("root", [r.value for r in RootNote], "root_id"),
        ("quality", [q.value for q in ChordQuality], "quality_id"),
        ("extension", [e.value for e in ChordExtension], "ext_id"),
    ):
        c = Counter(r[key] for r in all_chords if r[key] is not None)
        print(f"\n{head} head ({sum(c.values())} labeled):")
        for i, name in enumerate(names):
            n = c.get(i, 0)
            flag = "   <-- EMPTY" if n == 0 else ("   <-- rare" if n < 50 else "")
            print(f"  {i:>2} {name:<8} {n:>7}{flag}")


def write_split(split: str, sections: list[dict], chords_by_song: dict[str, list[dict]],
                out: Path, root: Path) -> int:
    split_sections = [s for s in sections if s["split"] == split]
    if not split_sections:
        return 0
    split_dir = out / split
    audio_dir = split_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    all_chords = []
    for sec in split_sections:
        src = root / sec["audio_path"]
        if src.exists():
            shutil.copy2(str(src), str(audio_dir / f"{sec['song_id']}.npy"))
        for chord in chords_by_song.get(sec["song_id"], []):
            all_chords.append(chord)

    if all_chords:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            table = {c: [r[c] for r in all_chords] for c in CHORD_COLUMNS}
            pq.write_table(pa.table(table), split_dir / "chords.parquet")
        except ImportError:
            import csv
            with open(split_dir / "chords.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(CHORD_COLUMNS)
                w.writerows([r[c] for c in CHORD_COLUMNS] for r in all_chords)
            print("  (pyarrow missing — wrote CSV instead of parquet)")

    return len(split_sections)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/export/v2")
    ap.add_argument("--only-ok", action="store_true")
    ap.add_argument("--keep-invalid-combos", action="store_true")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    statuses = ("ok",) if args.only_ok else ("ok", "partial")
    sections = fetch_sections(statuses)
    if not sections:
        print("training_sections has no rows")
        return

    song_ids = [s["song_id"] for s in sections]
    chords_by_song = fetch_chords(song_ids, statuses)

    all_chords = [c for cs in chords_by_song.values() for c in cs]
    if not args.keep_invalid_combos:
        all_chords = drop_invalid_combos(all_chords)
        valid_song_ids = {c["song_id"] for c in all_chords}
        chords_by_song = defaultdict(list)
        for c in all_chords:
            chords_by_song[c["song_id"]].append(c)

    print_distribution(sections, all_chords)
    if args.stats:
        return

    out = Path(args.out)
    root = training_v2_dir()
    counts = {s: write_split(s, sections, chords_by_song, out, root) for s in ("train", "val", "test")}

    config = {
        "sample_rate": 24000,
        "crop_s": 5,
        "dtype": "int16",
        "mono": True,
        "variable_length": True,
        "label_status_included": list(statuses),
        "pipeline_version": 2,
    }
    (out / "config.yaml").write_text(
        "\n".join(f"{k}: {json.dumps(v)}" for k, v in config.items()) + "\n",
        encoding="utf-8",
    )

    (out / "classes.json").write_text(json.dumps({
        "root":      [r.value for r in RootNote],
        "quality":   [q.value for q in ChordQuality],
        "extension": [e.value for e in ChordExtension],
        "valid_combo_mask": valid_combo_mask(),
        "counts": counts,
    }, indent=2), encoding="utf-8")
    print(f"\nExported {counts} → {out}")


if __name__ == "__main__":
    main()

