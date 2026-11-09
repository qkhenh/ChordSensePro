"""Measure label/audio time alignment of the v1 training data with chroma (diagnostic, read-only).

Why
  Listening checks (2026-10-27) found many Hooktheory segments whose label belongs to the chord
  *before* or *after* the audio. This script estimates, per song, by how many seconds the annotation
  timeline is shifted relative to the audio, WITHOUT downloading anything: it rebuilds the song's
  (separated-stem) audio from the stored 2 s segments, computes chroma, and slides the annotation
  timeline over it.

How
  1. Rebuild the song audio at 24 kHz from its ``training_samples`` rows (segment i is placed at
     ``start_ms``; gaps stay empty and are masked out).
  2. Chroma (CQT, hop 2048 = 85 ms) of the rebuilt audio, L2-normalised per frame.
  3. Annotation timeline -> one 12-d pitch-class template per chord (root + quality, 7th when known;
     9/11/13 and altered tones are ignored because chroma cannot confirm them).
  4. For every shift ``s`` in [-2.5 s, +2.5 s] (step 0.1 s): score = mean cosine(chroma(t), template(chord
     annotated at time t + s)). ``best_s`` is the shift with the highest score.

Sign convention
  annotation_time = audio_time + s.  s > 0  => the chord you hear at audio time t is written in the
  annotation at t + s (annotation runs *late*, labels must be read from t + s).  The fix is therefore
  to re-read the labels at ``t + best_s``.

Run inside the container (needs DB + .npy + librosa; scripts/ is bind-mounted):
    docker exec -w /opt/airflow chordsense_airflow env PYTHONPATH=/opt/airflow \\
        python scripts/measure_alignment.py --source hooktheory --limit 100
    # -> /data/export/align_report/<name>.csv  (host: DE/data/export/align_report/<name>.csv)

Sanity check of the method (known shift must be found back; sign/units are right):
    ... --source hooktheory --limit 30 --inject-shift 1.0
"""
from __future__ import annotations
import argparse
import csv
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/opt/airflow")

import numpy as np  # noqa: E402

from src.data_processing.application.annotation_parsers.parsers import get_parser  # noqa: E402
from src.data_processing.application.mappers.harte_mapper import map_harte           # noqa: E402
from src.data_processing.application.pipeline_handlers.sample_writer_handler import (  # noqa: E402
    training_dir,
)
from src.shared.infrastructure.postgres.client import get_connection                # noqa: E402

SR = 24_000
HOP = 2048
FRAME_SEC = HOP / SR
SHIFTS = np.round(np.arange(-2.5, 2.5001, 0.1), 2)
MIN_FRAMES = 30                      # ~2.5 s of usable evidence

# quality_id (enum order): 0 major, 1 minor, 2 dominant, 3 dim, 4 sus2, 5 sus4
TRIAD = {0: (0, 4, 7), 1: (0, 3, 7), 2: (0, 4, 7, 10), 3: (0, 3, 6), 4: (0, 2, 7), 5: (0, 5, 7)}
# ext_id (enum order): 3 maj7, 4 min7, 5 dom7 ; every other extension is ignored (chroma cannot confirm it)
SEVENTH = {3: 11, 4: 10, 5: 10}


def chord_template(root_id: int, quality_id: int, ext_id: int | None) -> np.ndarray | None:
    pcs = set(TRIAD[quality_id])
    if ext_id in SEVENTH and quality_id != 2:
        pcs.add(9 if (quality_id == 3 and ext_id == 5) else SEVENTH[ext_id])   # dim + dom7 = dim7 (b7b = 9)
    v = np.zeros(12, dtype=np.float32)
    for p in pcs:
        v[(root_id + p) % 12] = 1.0
    return v / np.linalg.norm(v)


def timeline_templates(timeline: list[tuple[float, float, str]]):
    """Timeline -> (starts, ends, templates) keeping only chords with a usable root and quality."""
    starts, ends, temps = [], [], []
    for start, end, label in timeline:
        m = map_harte(label)
        if m.root_id is None or m.quality_id is None:
            continue
        starts.append(start)
        ends.append(end)
        temps.append(chord_template(m.root_id, m.quality_id, m.ext_id))
    if not temps:
        return None
    return np.asarray(starts), np.asarray(ends), np.stack(temps)


def rebuild_audio(rows: list[dict], root: Path):
    end_sample = max(r["end_ms"] for r in rows) * SR // 1000 + 1
    y = np.zeros(end_sample, dtype=np.float32)
    have = np.zeros(end_sample, dtype=bool)
    cache: dict[str, np.ndarray] = {}
    for r in rows:
        arr = cache.get(r["audio_path"])
        if arr is None:
            arr = cache[r["audio_path"]] = np.load(root / r["audio_path"], mmap_mode="r")
        seg = np.asarray(arr[r["audio_idx"]], dtype=np.float32) / 32768.0
        a = r["start_ms"] * SR // 1000
        b = min(a + len(seg), end_sample)
        y[a:b] = seg[: b - a]
        have[a:b] = True
    return y, have


def song_scores(rows: list[dict], root: Path, inject: float):
    import librosa

    y, have = rebuild_audio(rows, root)
    if len(y) < SR * 3:
        return None
    parser = get_parser(Path(rows[0]["annotation_url"]))
    timeline = parser.parse(Path(rows[0]["annotation_url"]))
    if inject:
        timeline = [(s + inject, e + inject, lab) for s, e, lab in timeline]
    tt = timeline_templates(timeline)
    if tt is None:
        return None
    starts, ends, temps = tt

    chroma = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP)
    n = chroma.shape[1]
    norm = np.linalg.norm(chroma, axis=0)
    t = np.arange(n) * FRAME_SEC
    cover = np.array([have[min(int(x * SR), len(have) - 1)] for x in t])
    frame_ok = cover & (norm > 1e-3)
    if frame_ok.sum() < MIN_FRAMES:
        return None
    chroma = chroma / np.maximum(norm, 1e-6)
    sims = temps @ chroma                                   # (n_chords, n_frames)

    scores = []
    for s in SHIFTS:
        idx = np.searchsorted(starts, t + s, side="right") - 1
        valid = (idx >= 0) & frame_ok
        idx_c = np.clip(idx, 0, len(starts) - 1)
        valid &= (t + s) < ends[idx_c]
        if valid.sum() < MIN_FRAMES:
            scores.append(np.nan)
            continue
        scores.append(float(sims[idx_c[valid], np.nonzero(valid)[0]].mean()))
    scores = np.asarray(scores)
    if np.all(np.isnan(scores)):
        return None
    return scores, int(frame_ok.sum()), len(starts), len(y) / SR


def fetch(source: str, limit: int, seed: str):
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT song_id FROM training_samples WHERE source_type = %s "
            "GROUP BY song_id ORDER BY md5(song_id || %s) LIMIT %s", (source, seed, limit),
        )
        ids = [r[0] for r in cur.fetchall()]
        cur.execute(
            "SELECT ts.song_id, cq.annotation_url, ts.segment_idx, ts.start_ms, ts.end_ms, ts.audio_path, ts.audio_idx "
            "FROM training_samples ts JOIN crawl_queue cq ON cq.source_url = ts.source_url "
            "WHERE ts.song_id = ANY(%s) ORDER BY ts.song_id, ts.segment_idx", (ids,),
        )
        keys = ("song_id", "annotation_url", "segment_idx", "start_ms", "end_ms", "audio_path", "audio_idx")
        by_song: dict[str, list[dict]] = defaultdict(list)
        for r in cur.fetchall():
            by_song[r[0]].append(dict(zip(keys, r)))
    return by_song


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))] if values else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="hooktheory", help="hooktheory | jaah")
    p.add_argument("--limit", type=int, default=100, help="number of songs (random)")
    p.add_argument("--seed", default="0")
    p.add_argument("--inject-shift", type=float, default=0.0, dest="inject",
                   help="shift the timeline by this many seconds first (sanity check)")
    p.add_argument("--name", default=None)
    p.add_argument("--out", default="/data/export/align_report")
    args = p.parse_args()

    root = training_dir()
    by_song = fetch(args.source, args.limit, args.seed)
    name = args.name or f"{args.source}_{args.limit}" + (f"_inject{args.inject:+g}" if args.inject else "")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows_out, t0 = [], time.time()
    for i, (song, rows) in enumerate(by_song.items(), 1):
        try:
            res = song_scores(rows, root, args.inject)
        except Exception as e:                                   # one bad song must not stop the run
            print(f"[{i}/{len(by_song)}] {song[:8]} ERROR {e}")
            continue
        if res is None:
            continue
        scores, frames, n_chords, dur = res
        k0 = int(np.where(SHIFTS == 0.0)[0][0])
        best = int(np.nanargmax(scores))
        rows_out.append({
            "song_id": song, "source": args.source, "annotation": rows[0]["annotation_url"].split("/")[-1],
            "duration_s": round(dur, 1), "frames": frames, "n_chords": n_chords,
            "score_0": round(float(scores[k0]), 4) if not np.isnan(scores[k0]) else "",
            "best_shift_s": float(SHIFTS[best]), "best_score": round(float(scores[best]), 4),
            "median_score": round(float(np.nanmedian(scores)), 4),
        })
        if i % 20 == 0:
            print(f"  {i}/{len(by_song)} songs, {(time.time() - t0) / i:.1f} s/song")

    csv_path = out / f"{name}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    shifts = [r["best_shift_s"] for r in rows_out]
    best = [r["best_score"] for r in rows_out]
    med = [r["median_score"] for r in rows_out]
    gain = [r["best_score"] - r["score_0"] for r in rows_out if r["score_0"] != ""]
    print(f"\n=== {name}: {len(rows_out)} songs scored ({time.time() - t0:.0f} s) -> {csv_path}")
    print(f"best_shift_s : median {statistics.median(shifts):+.2f} | p10 {pct(shifts, .1):+.2f} | p90 {pct(shifts, .9):+.2f}")
    print(f"best_score   : median {statistics.median(best):.3f} | p10 {pct(best, .1):.3f} | p90 {pct(best, .9):.3f}")
    print(f"median_score (chance level over all shifts): {statistics.median(med):.3f}")
    print(f"gain over shift 0 : median {statistics.median(gain):.3f} | p90 {pct(gain, .9):.3f}")
    hist: dict[str, int] = defaultdict(int)
    for s in shifts:
        a = abs(s)
        hist["|s| <= 0.2" if a <= 0.2 else "0.2 < |s| <= 0.5" if a <= 0.5 else "0.5 < |s| <= 1.0" if a <= 1.0
             else "1.0 < |s| <= 2.0" if a <= 2.0 else "|s| > 2.0 (edge of search)"] += 1
    for k in ("|s| <= 0.2", "0.2 < |s| <= 0.5", "0.5 < |s| <= 1.0", "1.0 < |s| <= 2.0", "|s| > 2.0 (edge of search)"):
        print(f"  {k:28s} {hist[k]:4d}  ({hist[k] / len(shifts):.0%})")


if __name__ == "__main__":
    main()
