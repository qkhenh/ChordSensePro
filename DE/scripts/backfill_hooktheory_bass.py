"""Backfill Hooktheory slash-chord inversions (`label_raw` & `bass_id`) across DB and export v2/v3.

Usage:
    python DE/scripts/backfill_hooktheory_bass.py --dry-run
    python DE/scripts/backfill_hooktheory_bass.py
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DE_DIR = ROOT_DIR / "DE"
sys.path.insert(0, str(DE_DIR))

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from src.data_processing.application.annotation_parsers.hooktheory import build_annotation
from src.data_processing.application.mappers.harte_mapper import (
    BASS_NONE_ID,
    _strip_bass,
    parse_bass_id,
)
from src.shared.domain.value_objects import RootNote

BASS_NAMES = [r.value for r in RootNote] + ["none"]


def get_dsn() -> str:
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "chordsense")
    user = os.getenv("POSTGRES_USER", "chordsense")
    pw = os.getenv("POSTGRES_PASSWORD", "chordsense_dev")
    return f"postgresql://{user}:{pw}@{host}:{port}/{db}"


def rebuild_shifted_timeline(entry: dict, align_offset_ms: int) -> list[tuple[int, int, str]]:
    ann = build_annotation(entry, refined_only=True)
    if ann is None:
        ann = build_annotation(entry, refined_only=False)
    if ann is None:
        return []
    offset_sec = (align_offset_ms or 0) / 1000.0
    out: list[tuple[int, int, str]] = []
    for start, end, label in ann["timeline"]:
        adj_s = start - offset_sec
        adj_e = end - offset_sec
        if adj_e <= 0:
            continue
        adj_s = max(0.0, adj_s)
        out.append((round(adj_s * 1000), round(adj_e * 1000), label))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill Hooktheory bass_id into Postgres and v2/v3 parquets")
    ap.add_argument("--ht-gz", default=str(DE_DIR / "data/datasets/hooktheory/Hooktheory.json.gz"))
    ap.add_argument("--v2-dir", default=str(DE_DIR / "data/export/v2"))
    ap.add_argument("--v3-dir", default=str(DE_DIR / "data/export/v3"))
    ap.add_argument("--update-ann-files", action="store_true", help="Also refresh cached hooktheory/annotations/*.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ht_gz = Path(args.ht_gz)
    if not ht_gz.exists():
        raise FileNotFoundError(f"Hooktheory archive not found: {ht_gz}")

    print(f"Loading {ht_gz} ...")
    with gzip.open(ht_gz, "rt", encoding="utf-8") as f:
        ht_data: dict[str, dict] = json.load(f)
    print(f"Loaded {len(ht_data):,} Hooktheory entries.")

    conn = psycopg2.connect(get_dsn())
    try:
        with conn.cursor() as cur:
            if not args.dry_run:
                cur.execute("ALTER TABLE training_chords ADD COLUMN IF NOT EXISTS bass_id SMALLINT DEFAULT 12;")
                conn.commit()

            cur.execute(
                "SELECT song_id, source_type, source_url, align_offset_ms FROM training_sections"
            )
            sections = cur.fetchall()

            cur.execute(
                "SELECT id, song_id, start_ms, end_ms, label_raw, root_id "
                "FROM training_chords ORDER BY song_id, id"
            )
            chords_by_song: dict[str, list[tuple]] = defaultdict(list)
            for row in cur.fetchall():
                chords_by_song[row[1]].append(row)

        db_updates: list[tuple[int, str, int]] = []  # (id, label_raw, bass_id)
        parquet_lookup: dict[tuple[str, int, int, str], tuple[str, int]] = {}

        ht_matched_songs = 0
        ht_unmatched_songs = 0
        ht_slash_added = 0
        other_slash_found = 0

        for song_id, source_type, source_url, align_offset_ms in sections:
            db_rows = chords_by_song.get(song_id, [])
            if not db_rows:
                continue

            if source_type == "hooktheory" and "#ht=" in source_url:
                clip_id = source_url.split("#ht=", 1)[1]
                entry = ht_data.get(clip_id)
                if entry is None:
                    ht_unmatched_songs += 1
                    continue

                rebuilt = rebuild_shifted_timeline(entry, align_offset_ms or 0)
                if len(rebuilt) == len(db_rows) and all(
                    (r[0], r[1]) == (d[2], d[3]) for r, d in zip(rebuilt, db_rows)
                ):
                    ht_matched_songs += 1
                    for (s_ms, e_ms, new_label), (cid, _, _, _, old_label, root_id) in zip(rebuilt, db_rows):
                        assert _strip_bass(new_label) == _strip_bass(old_label), (
                            f"Base label mismatch on {song_id}: {new_label} vs {old_label}"
                        )
                        rid = int(root_id) if root_id is not None else None
                        new_bass = parse_bass_id(new_label, rid)
                        if new_bass != BASS_NONE_ID:
                            ht_slash_added += 1
                        db_updates.append((cid, new_label, new_bass))
                        parquet_lookup[(song_id, s_ms, e_ms, _strip_bass(old_label))] = (new_label, new_bass)
                else:
                    # Fallback lookup by (start_ms, end_ms, base_label)
                    by_span = {(s_ms, e_ms, _strip_bass(lbl)): lbl for s_ms, e_ms, lbl in rebuilt}
                    matched_any = False
                    for cid, _, s_ms, e_ms, old_label, root_id in db_rows:
                        base_old = _strip_bass(old_label)
                        new_label = by_span.get((s_ms, e_ms, base_old), old_label)
                        rid = int(root_id) if root_id is not None else None
                        new_bass = parse_bass_id(new_label, rid)
                        if new_bass != BASS_NONE_ID:
                            ht_slash_added += 1
                        if new_label != old_label:
                            matched_any = True
                        db_updates.append((cid, new_label, new_bass))
                        parquet_lookup[(song_id, s_ms, e_ms, base_old)] = (new_label, new_bass)
                    if matched_any or rebuilt:
                        ht_matched_songs += 1
                    else:
                        ht_unmatched_songs += 1
            else:
                for cid, _, s_ms, e_ms, old_label, root_id in db_rows:
                    rid = int(root_id) if root_id is not None else None
                    new_bass = parse_bass_id(old_label, rid)
                    if new_bass != BASS_NONE_ID:
                        other_slash_found += 1
                    db_updates.append((cid, old_label, new_bass))
                    parquet_lookup[(song_id, s_ms, e_ms, _strip_bass(old_label))] = (old_label, new_bass)

        print(
            f"Hooktheory songs matched: {ht_matched_songs:,} (unmatched: {ht_unmatched_songs}) | "
            f"Hooktheory slash chords rescued in DB: {ht_slash_added:,} | "
            f"Non-Hooktheory slash chords in DB: {other_slash_found:,}"
        )

        if not args.dry_run and db_updates:
            print(f"Updating {len(db_updates):,} rows in Postgres training_chords ...")
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    """
                    UPDATE training_chords AS tc
                    SET label_raw = v.label_raw,
                        bass_id   = v.bass_id
                    FROM (VALUES %s) AS v(id, label_raw, bass_id)
                    WHERE tc.id = v.id
                    """,
                    db_updates,
                    template="(%s::bigint, %s::varchar, %s::smallint)",
                    page_size=5000,
                )
            conn.commit()
            print("Postgres update committed.")
    finally:
        conn.close()

    # Optional: refresh cached annotation JSON files on disk if requested
    ann_dir = ht_gz.parent / "annotations"
    if args.update_ann_files and ann_dir.exists() and not args.dry_run:
        refreshed = 0
        for jf in ann_dir.glob("*.json"):
            entry = ht_data.get(jf.stem)
            if entry is None:
                continue
            ann = build_annotation(entry)
            if ann is not None:
                jf.write_text(json.dumps(ann, ensure_ascii=False), encoding="utf-8")
                refreshed += 1
        print(f"Refreshed {refreshed:,} Hooktheory annotation JSON files in {ann_dir}")

    # Update v2 & v3 parquets while preserving exact row order and schema
    v2_dir = Path(args.v2_dir)
    v3_dir = Path(args.v3_dir)
    bass_counter: Counter[int] = Counter()
    split_counts: dict[str, int] = {}
    total_v3_rows = 0

    for split in ("train", "val", "test"):
        v3_pq = v3_dir / split / "chords.parquet"
        if not v3_pq.exists():
            raise FileNotFoundError(f"Missing {v3_pq}")
        df = pd.read_parquet(v3_pq)
        new_labels: list[str] = []
        new_basses: list[int] = []

        for sid, s_ms, e_ms, raw, rid in zip(
            df["song_id"], df["start_ms"], df["end_ms"], df["label_raw"], df["root_id"]
        ):
            raw_str = str(raw) if pd.notna(raw) else ""
            base_raw = _strip_bass(raw_str)
            key = (str(sid), int(s_ms), int(e_ms), base_raw)
            if key in parquet_lookup:
                nlbl, nbass = parquet_lookup[key]
            else:
                nlbl = raw_str
                nbass = parse_bass_id(nlbl, int(rid) if pd.notna(rid) else None)
            new_labels.append(nlbl)
            new_basses.append(nbass)

        df["label_raw"] = new_labels
        df["bass_id"] = pd.array(new_basses, dtype="int64")

        bass_counter.update(new_basses)
        total_v3_rows += len(df)
        split_counts[split] = int(df["song_id"].nunique())

        slash_split = sum(1 for b in new_basses if b != BASS_NONE_ID)
        print(
            f"[{split}] {len(df):,} chords ({split_counts[split]:,} songs) -> "
            f"slash chords: {slash_split:,} ({100.0 * slash_split / max(len(df), 1):.2f}%)"
        )

        if not args.dry_run:
            df.to_parquet(v3_pq, index=False)
            v2_pq = v2_dir / split / "chords.parquet"
            if v2_pq.exists():
                df_v2 = df.drop(columns=["bass_id"]) if "bass_id" in df.columns else df
                df_v2.to_parquet(v2_pq, index=False)

    slash_total = sum(n for bid, n in bass_counter.items() if bid != BASS_NONE_ID)
    print(
        f"\nTotal v3 chords: {total_v3_rows:,} | "
        f"Total slash chords (bass != root): {slash_total:,} "
        f"({100.0 * slash_total / max(total_v3_rows, 1):.2f}%)"
    )
    print("\nBass head distribution (13 classes):")
    for i, name in enumerate(BASS_NAMES):
        n = bass_counter.get(i, 0)
        pct = 100.0 * n / max(total_v3_rows, 1)
        print(f"  {i:>2} {name:<6} {n:>8,} ({pct:5.2f}%)")

    classes_path = v3_dir / "classes.json"
    if not args.dry_run and classes_path.exists():
        meta = json.loads(classes_path.read_text(encoding="utf-8"))
        meta["bass_distribution"] = {str(i): int(bass_counter.get(i, 0)) for i in range(len(BASS_NAMES))}
        meta["counts"] = split_counts
        classes_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"\nUpdated {classes_path} and v2/v3 chords.parquet successfully.")


if __name__ == "__main__":
    main()

