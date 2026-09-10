"""seed_local_datasets.py — Insert local dataset files into crawl_queue.

Actual folder structure (after inspecting downloaded data):

  data/datasets/
    jaah/
      JAAH-v0.1/
        MTG-JAAH-7686b91/
          annotations/   ← 113 JSON files (artist+title+chord annotations)
                            NO audio — uses YouTube search via yt-dlp

    kaggle/
      archive/
        piano_triads/    ← WAV files named {note}_{quality}_{octave}_{variant}.wav
                            Audio IS here. Chord label = filename.

    choco/
      choco/
        choco/
          partitions/    ← JAMS files per track, YouTube IDs inside

    mcgill/              ← Download separately

Usage:
    pip install psycopg2-binary
    python scripts/seed_local_datasets.py --source jaah --dry-run
    python scripts/seed_local_datasets.py --source jaah
    python scripts/seed_local_datasets.py --source kaggle
    python scripts/seed_local_datasets.py --source choco
    python scripts/seed_local_datasets.py           # all sources
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

# In Docker: DATA_DIR=/data (env var set in docker-compose.yml)
# Local dev:  defaults to ./data relative to project root
DATA_DIR = Path(os.getenv("DATA_DIR", str(Path(__file__).parent.parent / "data")))
BASE_DIR = DATA_DIR / "datasets"

# Build DSN from individual env vars (set by docker-compose.yml)
# Falls back to localhost defaults for running outside Docker
_pg_host = os.getenv("POSTGRES_HOST", "localhost")
_pg_port = os.getenv("POSTGRES_PORT", "5432")
_pg_db   = os.getenv("POSTGRES_DB",   "chordsense")
_pg_user = os.getenv("POSTGRES_USER", "chordsense")
_pg_pass = os.getenv("POSTGRES_PASSWORD", "chordsense_dev")
PG_DSN   = f"postgresql://{_pg_user}:{_pg_pass}@{_pg_host}:{_pg_port}/{_pg_db}"

# ── Dataset scanners ──────────────────────────────────────────────────────────

def scan_jaah(base: Path) -> list[dict]:
    """Scan JAAH annotations.

    JAAH audio is NOT public — uses YouTube search via yt-dlp.
    source_url format: "ytsearch1:{artist} {title} jazz"
    JaahDownloader handles this with yt-dlp ytsearch.
    annotation_url: container path to local JSON file.
    """
    annot_dir = base / "jaah" / "JAAH-v0.1" / "MTG-JAAH-7686b91" / "annotations"
    if not annot_dir.exists():
        print(f"  [jaah] Not found: {annot_dir}")
        return []

    entries = []
    for f in sorted(annot_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            artist = data.get("artist", "").strip()
            title  = data.get("title",  "").strip()
            if not artist or not title:
                continue

            # YouTube search URL — JaahDownloader uses yt-dlp ytsearch
            source_url     = f"ytsearch1:{artist} {title} jazz"
            container_path = f"/data/datasets/jaah/JAAH-v0.1/MTG-JAAH-7686b91/annotations/{f.name}"

            entries.append({
                "source_url":     source_url,
                "annotation_url": container_path,
                "source_type":    "jaah",
                "display":        f"JAAH: {artist} — {title}",
            })
        except Exception as e:
            print(f"  [jaah] Skip {f.name}: {e}")

    print(f"  [jaah] Found {len(entries)} tracks")
    return entries


def scan_kaggle(base: Path) -> list[dict]:
    """Scan Kaggle piano_triads dataset.

    Files: archive/piano_triads/{note}_{quality}_{octave}_{variant}.wav
    Label is embedded in filename: A_maj_3_0.wav → chord = A:maj
    No annotation file needed — chord label stored as inline JSON.
    """
    triads_dir = base / "kaggle" / "archive" / "piano_triads"
    if not triads_dir.exists():
        print(f"  [kaggle] Not found: {triads_dir}")
        return []

    entries = []
    for wav in sorted(triads_dir.glob("*.wav")):
        parts = wav.stem.split("_")
        # Format: {note}_{quality}_{octave}_{variant}
        # note can be: A, B, Bb, C, Cs, D, Eb, E, F, Fs, G, Gs (1-2 parts)
        # quality: maj, min, dim (always 3 chars, after note)
        try:
            if len(parts) < 4:
                continue
            # Find quality index (maj/min/dim)
            qual_idx = next(
                (i for i, p in enumerate(parts) if p in ("maj", "min", "dim")), None
            )
            if qual_idx is None:
                continue

            note    = "_".join(parts[:qual_idx])   # e.g. "Bb", "Cs", "A"
            quality = parts[qual_idx]               # maj / min / dim

            # Map to standard chord notation
            qual_map = {"maj": "maj", "min": "min", "dim": "dim"}
            chord_label = f"{note}:{qual_map[quality]}"

            container_path = f"/data/datasets/kaggle/archive/piano_triads/{wav.name}"

            entries.append({
                "source_url":     container_path,
                "annotation_url": "",   # label in filename — no annotation file
                "source_type":    "local",
                "chord_label":    chord_label,  # stored as metadata
                "display":        f"Kaggle: {chord_label} ({wav.name})",
            })
        except Exception as e:
            print(f"  [kaggle] Skip {wav.name}: {e}")

    print(f"  [kaggle] Found {len(entries)} files")
    return entries


def scan_choco(base: Path, limit: int = 500) -> list[dict]:
    """Scan ChoCo partitions — reads YouTube ID from JAMS files.

    Actual path after clone: choco/choco/choco/partitions/
    """
    # Try multiple possible paths (nested git clone structures)
    candidates = [
        base / "choco" / "partitions",
        base / "choco" / "choco" / "partitions",
        base / "choco" / "choco" / "choco" / "partitions",
    ]
    partitions_dir = next((p for p in candidates if p.exists()), None)
    if not partitions_dir:
        print(f"  [choco] Partitions not found, tried: {[str(c) for c in candidates]}")
        return []

    print(f"  [choco] Using: {partitions_dir}")
    entries = []
    for jams_file in sorted(partitions_dir.rglob("ann_audio_chord.jams")):
        if limit and len(entries) >= limit:
            break
        try:
            data = json.loads(jams_file.read_text(encoding="utf-8"))
            meta = data.get("file_metadata", {})
            ids  = meta.get("identifiers", {})
            yt_id = ids.get("youtube_id") or ids.get("youtube") or ids.get("id_youtube")
            if not yt_id:
                continue

            # Compute container path relative to base
            try:
                rel = jams_file.relative_to(base.parent.parent.parent)  # relative to DE/
                container_annot = f"/data/datasets/{jams_file.relative_to(base)}".replace("\\", "/")
            except ValueError:
                container_annot = ""

            entries.append({
                "source_url":     f"https://www.youtube.com/watch?v={yt_id}",
                "annotation_url": container_annot,
                "source_type":    "choco",
                "display":        f"ChoCo: {meta.get('title', yt_id)} — {yt_id}",
            })
        except Exception as e:
            print(f"  [choco] Skip {jams_file.name}: {e}")

    print(f"  [choco] Found {len(entries)} tracks (limit={limit})")
    return entries


# ── Database insert ───────────────────────────────────────────────────────────

def seed_to_db(entries: list[dict], dry_run: bool = False) -> tuple[int, int]:
    """Insert entries into crawl_queue. Returns (inserted, skipped)."""
    if dry_run:
        print(f"\n[DRY RUN] Would insert up to {len(entries)} entries:")
        for e in entries[:10]:
            print(f"  {e['display']}")
        if len(entries) > 10:
            print(f"  ... and {len(entries) - 10} more")
        return 0, 0

    import psycopg2
    conn = psycopg2.connect(PG_DSN)
    conn.autocommit = False
    cur  = conn.cursor()

    inserted = skipped = 0
    now = datetime.now(timezone.utc)

    for entry in entries:
        if not entry.get("source_url"):
            skipped += 1
            continue
        try:
            cur.execute("""
                INSERT INTO crawl_queue
                    (id, source_url, annotation_url, source_type, status, priority, created_at)
                VALUES (%s, %s, %s, %s, 'pending', 0, %s)
                ON CONFLICT (source_url) DO NOTHING
            """, (
                str(uuid.uuid4()),
                entry["source_url"],
                entry.get("annotation_url", ""),
                entry["source_type"],
                now,
            ))
            inserted += 1 if cur.rowcount == 1 else 0
            skipped  += 1 if cur.rowcount == 0 else 0
        except Exception as e:
            print(f"  Insert error for {entry.get('display', '')}: {e}")
            skipped += 1

    conn.commit()
    cur.close()
    conn.close()
    return inserted, skipped


# ── Main ──────────────────────────────────────────────────────────────────────

SCANNERS = {
    "jaah":   scan_jaah,
    "kaggle": scan_kaggle,
    "choco":  lambda base: scan_choco(base, limit=500),
}


def main():
    parser = argparse.ArgumentParser(description="Seed crawl_queue from local dataset files")
    parser.add_argument("--source",  choices=list(SCANNERS) + ["all"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--data-dir", default=str(BASE_DIR))
    args = parser.parse_args()

    base = Path(args.data_dir)
    if not base.exists():
        print(f"ERROR: {base} not found")
        sys.exit(1)

    sources = list(SCANNERS.keys()) if args.source == "all" else [args.source]

    all_entries: list[dict] = []
    print(f"\nScanning: {base}\n")
    for src in sources:
        print(f"[{src}]")
        all_entries.extend(SCANNERS[src](base))

    if not all_entries:
        print("\nNo entries found. Check dataset_download_guide.md")
        sys.exit(0)

    print(f"\nTotal: {len(all_entries)} entries")
    inserted, skipped = seed_to_db(all_entries, dry_run=args.dry_run)

    if not args.dry_run:
        print(f"Done: {inserted} inserted, {skipped} already existed")
        print("Next: Trigger dag_ingest_raw in Airflow UI → localhost:8080")


if __name__ == "__main__":
    main()
