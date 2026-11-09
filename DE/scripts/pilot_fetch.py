"""P1b pilot, step 1: pick pilot songs and re-download their RAW audio (needs internet).

Context (DE v2 plan, phase P1b): compare vocal/drum/bass-removal models. The v1 pipeline deleted
its raw WAVs, so the pilot songs have to be fetched again.

Selection (random, seeded) from the alignment report written by ``measure_alignment.py``:
  * Hooktheory : |best_shift| <= 0.2 s and peak prominence >= 0.10   (labels we can trust)
  * JAAH       : |best_shift| <= 0.5 s
  Caveat: the filter used Demucs stems to judge the alignment, so it is slightly biased towards songs
  where Demucs already works. This only matters for the absolute scores, not for the model ranking
  on the same songs.

Output (host: DE/data/export/pilot/):
    raw/<song_id>.wav        44.1 kHz mono. JAAH is cut to a 60 s excerpt (60..120 s) to keep the pilot fast.
    manifest.json            [{song_id, source, annotation, offset_s, duration_s}]   offset_s: audio t=0 is
                             annotation time ``offset_s`` (0 for Hooktheory, 60 for the JAAH excerpt).

Run (inside the container):
    docker exec -w /opt/airflow chordsense_airflow env PYTHONPATH=/opt/airflow \\
        python scripts/pilot_fetch.py --n-hooktheory 40 --n-jaah 10
"""
from __future__ import annotations
import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "/opt/airflow")

import numpy as np          # noqa: E402
import soundfile as sf      # noqa: E402

from src.data_loader.application.entrypoints import run_data_loader                  # noqa: E402
from src.data_loader.domain.models.audio_file import AudioFile, AudioSource          # noqa: E402
from src.data_processing.application.annotation_parsers.parsers import (             # noqa: E402
    read_annotation_duration, read_annotation_section,
)
from src.shared.infrastructure.postgres.client import get_connection                 # noqa: E402

REPORT = Path("/data/export/align_report")
OUT = Path("/data/export/pilot")
JAAH_CROP = (60.0, 120.0)


def candidates(source: str, max_shift: float, min_prom: float | None) -> list[str]:
    path = REPORT / f"{source}_all.csv"
    ids = []
    for r in csv.DictReader(open(path)):
        if abs(float(r["best_shift_s"])) > max_shift:
            continue
        if min_prom is not None and float(r["best_score"]) - float(r["median_score"]) < min_prom:
            continue
        ids.append(r["song_id"])
    return ids


def lookup(song_ids: list[str]) -> dict[str, tuple[str, str]]:
    """song_id -> (source_url, annotation_path)."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT ts.song_id, ts.source_url, cq.annotation_url FROM training_samples ts "
            "JOIN crawl_queue cq ON cq.source_url = ts.source_url WHERE ts.song_id = ANY(%s)", (song_ids,))
        return {r[0]: (r[1], r[2]) for r in cur.fetchall()}


def fetch_one(song_id: str, source: str, url: str, ann: str, tmp: Path) -> dict | None:
    section = read_annotation_section(Path(ann))
    expected = read_annotation_duration(Path(ann))
    dest = tmp / song_id
    resp = run_data_loader(AudioFile.from_url(
        url, song_id, dest, source_type=AudioSource(source), dataset_name=song_id,
        expected_duration=expected, section=section))
    if not resp.is_ok or resp.wav_path is None:
        print(f"  FAIL {song_id[:8]}: {resp.error}")
        return None
    y, sr = sf.read(str(resp.wav_path), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    offset = 0.0
    if source == "jaah":
        a, b = int(JAAH_CROP[0] * sr), int(JAAH_CROP[1] * sr)
        if len(y) < b:
            print(f"  SKIP {song_id[:8]}: shorter than the crop window")
            return None
        y, offset = y[a:b], JAAH_CROP[0]
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    sf.write(str(OUT / "raw" / f"{song_id}.wav"), y, sr, subtype="PCM_16")
    return {"song_id": song_id, "source": source, "annotation": ann, "offset_s": offset,
            "duration_s": round(len(y) / sr, 2), "sr": sr}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-hooktheory", type=int, default=40)
    p.add_argument("--n-jaah", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    rng = random.Random(args.seed)
    plan = []
    for source, n, max_shift, prom in (("hooktheory", args.n_hooktheory, 0.2, 0.10),
                                       ("jaah", args.n_jaah, 0.5, None)):
        ids = candidates(source, max_shift, prom)
        rng.shuffle(ids)
        plan += [(source, i) for i in ids[: int(n * 1.6)]]      # extra, some videos are dead
        print(f"{source}: {len(ids)} candidates")
    info = lookup([i for _, i in plan])

    tmp = Path("/data/tmp/pilot_dl")
    manifest, got = [], {"hooktheory": 0, "jaah": 0}
    want = {"hooktheory": args.n_hooktheory, "jaah": args.n_jaah}
    for source, sid in plan:
        if got[source] >= want[source] or sid not in info:
            continue
        print(f"[{source} {got[source] + 1}/{want[source]}] {sid[:8]}")
        rec = fetch_one(sid, source, info[sid][0], info[sid][1], tmp)
        shutil.rmtree(tmp / sid, ignore_errors=True)
        if rec:
            manifest.append(rec)
            got[source] += 1
            (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("done:", got, "->", OUT / "manifest.json")


if __name__ == "__main__":
    main()
