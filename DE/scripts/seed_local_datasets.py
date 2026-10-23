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
    python scripts/seed_local_datasets.py --source hooktheory --hooktheory-scope rare   # default
    python scripts/seed_local_datasets.py --source hooktheory --hooktheory-scope all
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


def scan_choco(base: Path, limit: int = 0) -> list[dict]:
    """Scan ChoCo partitions that carry a YouTube link.

    Real layout (git clone smashub/choco):
        choco/choco/partitions/<partition>/choco/meta.csv
        choco/choco/partitions/<partition>/choco/jams/<id>.jams

    Only partitions whose meta.csv has a ``youtube_url`` column have audio we can
    fetch (currently ``chordify`` = Chordify Annotator Subjectivity Dataset).
    billboard / ireal-pro / biab-internet-corpus are annotation-only (no audio).
    Several annotators may point at the same video; ``source_url`` is UNIQUE in
    crawl_queue so the first annotator wins and the rest are skipped.
    """
    import csv

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
    entries: list[dict] = []
    seen: set[str] = set()
    for meta_csv in sorted(partitions_dir.glob("*/choco/meta.csv")):
        partition = meta_csv.parent.parent.name
        with meta_csv.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if "youtube_url" not in (reader.fieldnames or []):
                print(f"  [choco] {partition}: no youtube_url column → annotation-only, skipped")
                continue
            for row in reader:
                if limit and len(entries) >= limit:
                    break
                url = (row.get("youtube_url") or "").strip()
                tid = (row.get("id") or "").strip()
                if not url or not tid or url in seen:
                    continue
                jams_file = meta_csv.parent / "jams" / f"{tid}.jams"
                if not jams_file.exists():
                    continue
                seen.add(url)
                container_annot = f"/data/datasets/{jams_file.relative_to(base)}".replace("\\", "/")
                entries.append({
                    "source_url":     url,
                    "annotation_url": container_annot,
                    "source_type":    "choco",
                    "display":        f"ChoCo/{partition}: {row.get('track_title', tid)} — {row.get('track_performer', '')}",
                })

    print(f"  [choco] Found {len(entries)} tracks with audio link")
    return entries


def _container_path(path: Path, base: Path) -> str:
    """Host path under data/datasets → path as seen inside the container (/data/datasets/...)."""
    return f"/data/datasets/{path.relative_to(base).as_posix()}"


def scan_guitarset(base: Path) -> list[dict]:
    """GuitarSet — 360 acoustic-guitar clips (~30 s) with JAMS chords.

    Layout:  guitarset/annotation/<id>.jams   guitarset/audio_mono-mic/<id>_mic.wav
    JamsParser prefers the 'performed' chord track (rich: sus2/sus4, maj6, omitted notes).
    Fast to process (clean source: no Demucs) and the main source of *suspended* chords.
    """
    root = base / "guitarset"
    ann_dir, audio_dir = root / "annotation", root / "audio_mono-mic"
    if not ann_dir.exists() or not audio_dir.exists():
        print(f"  [guitarset] Not found under {root}")
        return []

    entries = []
    for jams in sorted(ann_dir.glob("*.jams")):
        wav = audio_dir / f"{jams.stem}_mic.wav"
        if not wav.exists():
            continue
        entries.append({
            "source_url":     _container_path(wav, base),
            "annotation_url": _container_path(jams, base),
            "source_type":    "guitarset",
            "priority":       5,
            "display":        f"GuitarSet: {jams.stem}",
        })
    print(f"  [guitarset] Found {len(entries)} clips")
    return entries


def scan_swd(base: Path) -> list[dict]:
    """Schubert Winterreise Dataset — only the versions whose audio is shipped (HU33, SC06).

    Layout:  <swd>/01_RawData/audio_wav/Schubert_D911-01_HU33.wav
             <swd>/02_Annotations/ann_audio_chord/Schubert_D911-01_HU33.csv  (the other 7 versions
             have annotations but no audio, so they are skipped)
    """
    roots = sorted((base / "Schubert_Winterreise").glob("*/01_RawData")) \
        or sorted(base.glob("Schubert_Winterreise*/**/01_RawData"))
    if not roots:
        print(f"  [swd] Not found under {base / 'Schubert_Winterreise'}")
        return []
    raw = roots[0]
    ann_dir = raw.parent / "02_Annotations" / "ann_audio_chord"

    entries = []
    for wav in sorted((raw / "audio_wav").glob("*.wav")):
        csv_file = ann_dir / f"{wav.stem}.csv"
        if not csv_file.exists():
            continue
        entries.append({
            "source_url":     _container_path(wav, base),
            "annotation_url": _container_path(csv_file, base),
            "source_type":    "swd",
            "priority":       4,
            "display":        f"SWD: {wav.stem}",
        })
    print(f"  [swd] Found {len(entries)} recordings")
    return entries


# ChoCo-release audio partitions and their queue priority. Annotation-only / duplicate partitions
# are deliberately absent: jaah (already seeded from the JAAH repo), casd (chordify, seeded by
# scan_choco), schubert-winterreise-audio (local SWD audio is used instead), weimar (solo-only
# excerpts, duration would not match a full YouTube track), and all symbolic partitions.
CHOCO_RELEASE_PARTITIONS = {
    "isophonics":      3,
    "robbie-williams": 3,
    "rwc-pop":         3,
    "uspop2002":       3,
    "billboard":       1,
}


def scan_choco_release(base: Path) -> list[dict]:
    """ChoCo GitHub release (v1.0.0.zip) — audio partitions that only carry title/performer/duration.

    Audio is found via 'ytsearch1:<performer> <title>' and the candidate whose length matches the
    JAMS ``file_metadata.duration`` is downloaded (ChocoDownloader → JAAH search logic); wrong
    versions are rejected by the duration check.
    """
    jams_dirs = sorted((base / "choco_release").glob("**/jams"))
    if not jams_dirs:
        print(f"  [choco-release] Not found under {base / 'choco_release'}")
        return []
    jams_dir = jams_dirs[0]

    entries: list[dict] = []
    seen: set[str] = set()
    for partition, priority in CHOCO_RELEASE_PARTITIONS.items():
        count = 0
        for jams_file in sorted(jams_dir.glob(f"{partition}_*.jams")):
            try:
                data = json.loads(jams_file.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"  [choco-release] Skip {jams_file.name}: {e}")
                continue
            sandbox = data.get("sandbox") or {}
            meta = data.get("file_metadata") or {}
            performers = [p for p in (sandbox.get("performers") or []) if p]
            title = (meta.get("title") or "").strip()
            if sandbox.get("type") != "audio" or not title or not performers or not meta.get("duration"):
                continue
            source_url = f"ytsearch1:{performers[0]} {title}"
            if source_url in seen:
                continue
            seen.add(source_url)
            count += 1
            entries.append({
                "source_url":     source_url,
                "annotation_url": _container_path(jams_file, base),
                "source_type":    "choco",
                "priority":       priority,
                "display":        f"ChoCo/{partition}: {performers[0]} — {title}",
            })
        print(f"  [choco-release] {partition}: {count} tracks")
    return entries


# ChordExtension values that make a chord "rare" in the class distribution (plain triads and
# 7th chords are already abundant). Suspended chords (quality sus2 / sus4) are rare as well.
_COMMON_EXTENSIONS = ("none", "maj7", "min7", "dom7")
HOOKTHEORY_SCOPE = "rare"          # "rare" | "all" — set from --hooktheory-scope


def scan_hooktheory(base: Path) -> list[dict]:
    """Hooktheory (Sheet Sage ``Hooktheory.json.gz``): ~20 s annotated YouTube clips.

    Layout:  hooktheory/Hooktheory.json.gz            (github.com/chrisdonahue/sheetsage-data)
             hooktheory/annotations/<id>.json         ← written here, one per clip
    Each clip becomes: source_url = https://www.youtube.com/watch?v=<id>#ht=<clip id>, plus an
    annotation JSON holding the section to download and the chords in clip time.

    scope ``rare`` seeds only clips with at least one suspended / extended / altered chord
    (priority 6, or 7 when they contain three or more); ``all`` also seeds the rest at priority 0.
    """
    import gzip
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data_processing.application.annotation_parsers.hooktheory import build_annotation
    from src.data_processing.application.mappers.harte_mapper import LabelStatus, map_harte

    root = base / "hooktheory"
    gz_path = root / "Hooktheory.json.gz"
    if not gz_path.exists():
        print(f"  [hooktheory] Not found: {gz_path}")
        return []
    ann_dir = root / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    with gzip.open(gz_path, "rt", encoding="utf-8") as fh:
        dataset = json.load(fh)

    entries: list[dict] = []
    skipped = rare_clips = 0
    for key, entry in dataset.items():
        ann = build_annotation(entry)
        if ann is None or not ann.get("id"):
            skipped += 1
            continue

        rare_events = 0
        for _, _, label in ann["timeline"]:
            mapped = map_harte(label)
            if mapped.status != LabelStatus.OK:
                continue
            ext = mapped.extension.name if mapped.extension is not None else "none"
            if (mapped.quality is not None and mapped.quality.name in ("sus2", "sus4")) or ext not in _COMMON_EXTENSIONS:
                rare_events += 1
        if rare_events:
            rare_clips += 1
        elif HOOKTHEORY_SCOPE != "all":
            continue

        ann_file = ann_dir / f"{ann['id']}.json"
        ann_file.write_text(json.dumps(ann, ensure_ascii=False), encoding="utf-8")
        entries.append({
            "source_url":     f"https://www.youtube.com/watch?v={ann['youtube_id']}#ht={ann['id']}",
            "annotation_url": _container_path(ann_file, base),
            "source_type":    "hooktheory",
            "priority":       (7 if rare_events >= 3 else 6) if rare_events else 0,
            "display":        f"Hooktheory: {ann['artist']} — {ann['song']} ({rare_events} rare chords)",
        })

    print(f"  [hooktheory] {len(entries)} clips seeded ({rare_clips} with rare chords; "
          f"{skipped} entries unusable; scope={HOOKTHEORY_SCOPE})")
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
                VALUES (%s, %s, %s, %s, 'pending', %s, %s)
                ON CONFLICT (source_url) DO NOTHING
            """, (
                str(uuid.uuid4()),
                entry["source_url"],
                entry.get("annotation_url", ""),
                entry["source_type"],
                entry.get("priority", 0),
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
    "choco":  scan_choco,
    "guitarset":     scan_guitarset,
    "swd":           scan_swd,
    "choco-release": scan_choco_release,
    "hooktheory":    scan_hooktheory,
}


def main():
    parser = argparse.ArgumentParser(description="Seed crawl_queue from local dataset files")
    parser.add_argument("--source",  choices=list(SCANNERS) + ["all"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--data-dir", default=str(BASE_DIR))
    parser.add_argument("--hooktheory-scope", choices=["rare", "all"], default="rare",
                        help="rare = only clips with sus/extended/altered chords (default); all = every usable clip")
    args = parser.parse_args()
    global HOOKTHEORY_SCOPE
    HOOKTHEORY_SCOPE = args.hooktheory_scope

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
        print("\nNo entries found. Check markdown/dataset_sources.md and setup.md section 2")
        sys.exit(0)

    print(f"\nTotal: {len(all_entries)} entries")
    inserted, skipped = seed_to_db(all_entries, dry_run=args.dry_run)

    if not args.dry_run:
        print(f"Done: {inserted} inserted, {skipped} already existed")
        print("Next: Trigger dag_ingest_raw in Airflow UI → localhost:8080")


if __name__ == "__main__":
    main()
