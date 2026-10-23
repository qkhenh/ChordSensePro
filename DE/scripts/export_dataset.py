"""Export the training dataset from PostgreSQL ``training_samples`` + the .npy audio files.

Output (one pair per split) — ready to upload to Colab / Drive:

    {out}/train_audio.npy   int16  (N, 48000)   2 s @ 24 kHz, mono   (np.load(..., mmap_mode='r'))
    {out}/train_meta.parquet        one row per segment, same order as the audio rows
    {out}/val_*  {out}/test_*  (same)
    {out}/classes.json      class-id → name tables + per-split counts

Meta columns: song_id, source_type, source_url, segment_idx, start_ms, end_ms, label_raw,
              label_status, root_id, quality_id, ext_id, split, rms_dbfs
              (rms_dbfs = loudness of the raw int16 segment, full scale = 0 dBFS; measured BEFORE any
              normalisation — use it to calibrate the inference-time silence gate)
Missing heads: for ``partial`` labels (e.g. 6th / power chords) quality_id and/or ext_id are null
→ the trainer must mask those heads in the loss.  Default export keeps status ok + partial;
use --only-ok for the strictest set.
Silent segments: segments quieter than --min-rms-dbfs (default -50 dBFS) are dropped — they carry a
chord label but contain (almost) no sound (usually a gap left by stem separation) = label noise.
Invalid combos: rows whose (quality, extension) pair is not in VALID_COMBOS (e.g. dim + b9) are dropped —
inference masks those pairs, so training on them would teach what the model may not predict.
Partial rows (a null head) cannot be judged and are kept.

Run inside the container (DB + files live there):
    docker exec chordsense_airflow python /opt/airflow/scripts/export_dataset.py
    # → /data/export  is bind-mounted to  DE/data/export  on the host

Options
  --out DIR            output directory (default /data/export)
  --only-ok            keep label_status = ok only
  --min-rms-dbfs DB    drop segments quieter than this (default -50)
  --keep-silent        do not drop quiet segments (rms_dbfs is still written)
  --keep-invalid-combos  do not drop rows with an invalid (quality, extension) pair
  --stats              print class distribution only, write nothing
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "/opt/airflow")

import numpy as np  # noqa: E402

from src.shared.domain.value_objects import (   # noqa: E402
    VALID_COMBOS, ChordExtension, ChordQuality, RootNote, valid_combo_mask,
)
from src.shared.infrastructure.postgres.client import get_connection                 # noqa: E402
from src.data_processing.application.pipeline_handlers.sample_writer_handler import (  # noqa: E402
    SEGMENT_LEN, training_dir,
)

COLUMNS = [
    "song_id", "source_type", "source_url", "segment_idx", "start_ms", "end_ms",
    "audio_path", "audio_idx", "label_raw", "label_status",
    "root_id", "quality_id", "ext_id", "split",
]
META_EXTRA = ["rms_dbfs"]          # computed from the audio at export time, not stored in the DB
DEFAULT_MIN_RMS_DBFS = -50.0


def fetch_rows(statuses: tuple[str, ...]) -> list[dict]:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT {', '.join(COLUMNS)} FROM training_samples "
            "WHERE label_status = ANY(%s) ORDER BY split, song_id, segment_idx",
            (list(statuses),),
        )
        return [dict(zip(COLUMNS, r)) for r in cur.fetchall()]


def attach_rms(rows: list[dict], root: Path) -> None:
    """Add ``rms_dbfs`` to every row (int16 full scale = 32768 → 0 dBFS). Reads one song at a time."""
    by_path: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_path[r["audio_path"]].append(r)
    for n, (path, group) in enumerate(by_path.items(), 1):
        arr = np.load(root / path, mmap_mode="r")
        seg = np.asarray(arr[[r["audio_idx"] for r in group]], dtype=np.float32)
        rms = np.sqrt((seg ** 2).mean(axis=1))
        db = 20.0 * np.log10(np.maximum(rms, 1e-9) / 32768.0)
        for r, d in zip(group, db):
            r["rms_dbfs"] = round(float(d), 1)
        if n % 500 == 0:
            print(f"  measured loudness of {n}/{len(by_path)} songs ...")


def drop_silent(rows: list[dict], min_db: float) -> list[dict]:
    kept = [r for r in rows if r["rms_dbfs"] >= min_db]
    dropped = [r for r in rows if r["rms_dbfs"] < min_db]
    pct = 100.0 * len(dropped) / max(len(rows), 1)
    print(f"\nsilent segments (RMS < {min_db:g} dBFS) dropped: {len(dropped)} of {len(rows)} ({pct:.2f}%)")
    if dropped:
        print("  per source:", dict(Counter(r["source_type"] for r in dropped)))
        print("  per split :", dict(Counter(r["split"] for r in dropped)))
    return kept


def print_distribution(rows: list[dict]) -> None:
    by_split = Counter(r["split"] for r in rows)
    print("\nsamples per split :", dict(by_split))
    print("per source        :", dict(Counter(r["source_type"] for r in rows)))
    songs = defaultdict(set)
    for r in rows:
        songs[r["split"]].add(r["song_id"])
    print("songs per split   :", {k: len(v) for k, v in songs.items()})
    for head, names, key in (
        ("root", [r.value for r in RootNote], "root_id"),
        ("quality", [q.value for q in ChordQuality], "quality_id"),
        ("extension", [e.value for e in ChordExtension], "ext_id"),
    ):
        c = Counter(r[key] for r in rows if r[key] is not None)
        print(f"\n{head} head ({sum(c.values())} labeled):")
        for i, name in enumerate(names):
            n = c.get(i, 0)
            flag = "   <-- EMPTY" if n == 0 else ("   <-- rare" if n < 50 else "")
            print(f"  {i:>2} {name:<8} {n:>7}{flag}")


def drop_invalid_combos(rows: list[dict]) -> list[dict]:
    """Drop rows whose (quality, extension) pair is not a real chord. Rows with a null head are kept."""
    qualities, exts = list(ChordQuality), list(ChordExtension)
    kept, dropped = [], []
    for r in rows:
        q, e = r["quality_id"], r["ext_id"]
        if q is not None and e is not None and (qualities[q], exts[e]) not in VALID_COMBOS:
            dropped.append(r)
        else:
            kept.append(r)
    pct = 100.0 * len(dropped) / max(len(rows), 1)
    print(f"\ninvalid (quality, extension) combos dropped: {len(dropped)} of {len(rows)} ({pct:.2f}%)")
    if dropped:
        top = Counter((qualities[r["quality_id"]].value, exts[r["ext_id"]].value) for r in dropped)
        print("  top:", dict(top.most_common(8)))
    return kept


def write_split(rows: list[dict], split: str, out: Path, root: Path) -> int:
    part = [r for r in rows if r["split"] == split]
    if not part:
        return 0
    out.mkdir(parents=True, exist_ok=True)
    audio = np.lib.format.open_memmap(
        out / f"{split}_audio.npy", mode="w+", dtype=np.int16, shape=(len(part), SEGMENT_LEN)
    )
    cache: dict[str, np.ndarray] = {}
    for i, r in enumerate(part):
        arr = cache.get(r["audio_path"])
        if arr is None:
            cache.clear()                                   # keep at most one song in RAM
            arr = cache[r["audio_path"]] = np.load(root / r["audio_path"], mmap_mode="r")
        audio[i] = arr[r["audio_idx"]]
    audio.flush()

    meta_cols = [c for c in COLUMNS if c not in ("audio_path", "audio_idx")] + META_EXTRA
    table = {c: [r[c] for r in part] for c in meta_cols}
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.table(table), out / f"{split}_meta.parquet")
    except ImportError:
        import csv
        with open(out / f"{split}_meta.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(meta_cols)
            w.writerows(zip(*[table[c] for c in meta_cols]))
        print("  (pyarrow missing — wrote CSV instead of parquet)")
    return len(part)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/data/export")
    ap.add_argument("--only-ok", action="store_true")
    ap.add_argument("--min-rms-dbfs", type=float, default=DEFAULT_MIN_RMS_DBFS)
    ap.add_argument("--keep-silent", action="store_true")
    ap.add_argument("--keep-invalid-combos", action="store_true")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    statuses = ("ok",) if args.only_ok else ("ok", "partial")
    rows = fetch_rows(statuses)
    if not rows:
        print("training_samples has no rows with status", statuses)
        return
    total_fetched = len(rows)
    if not args.keep_invalid_combos:
        rows = drop_invalid_combos(rows)
    dropped_invalid = total_fetched - len(rows)
    root = training_dir()
    print(f"measuring loudness of {len(rows)} segments ...")
    attach_rms(rows, root)
    before_silent = len(rows)
    if not args.keep_silent:
        rows = drop_silent(rows, args.min_rms_dbfs)
    print_distribution(rows)
    if args.stats:
        return

    out = Path(args.out)
    counts = {s: write_split(rows, s, out, root) for s in ("train", "val", "test")}
    (out / "classes.json").write_text(json.dumps({
        "root":      [r.value for r in RootNote],
        "quality":   [q.value for q in ChordQuality],
        "extension": [e.value for e in ChordExtension],
        "valid_combo_mask": valid_combo_mask(),      # [quality_id][ext_id] -> bool, for inference masking
        "sample_rate": 24000, "segment_samples": SEGMENT_LEN, "dtype": "int16",
        "label_status_included": list(statuses), "counts": counts,
        "min_rms_dbfs": None if args.keep_silent else args.min_rms_dbfs,
        "dropped_silent": 0 if args.keep_silent else before_silent - len(rows),
        "dropped_invalid_combos": dropped_invalid,
    }, indent=2), encoding="utf-8")
    print(f"\nExported {counts} → {out}")


if __name__ == "__main__":
    main()
