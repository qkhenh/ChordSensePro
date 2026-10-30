"""P1b pilot, step 2: remove drums / vocals / bass with several separators, then score the result.

Run with the pilot venv (it has ``audio-separator``; it also sees the container's torch + CUDA):

    docker exec -w /opt/airflow chordsense_airflow env PYTHONPATH=/opt/airflow \\
        /home/airflow/pilot_venv/bin/python scripts/pilot_separate.py run --variants demucs6s demucs_ft bsrof_sw chain
    docker exec -w /opt/airflow chordsense_airflow env PYTHONPATH=/opt/airflow \\
        /home/airflow/pilot_venv/bin/python scripts/pilot_separate.py score

Variants (the "training stem" = everything except drums, bass and vocals):
    mix        control, the raw full mix (no separation)
    demucs6s   v1 baseline  : htdemucs_6s   -> guitar + piano + other
    demucs_ft  htdemucs_ft (4 stems)        -> other
    bsrof_sw   BS-Roformer-SW (6 stems)     -> guitar + piano + other
    chain      BS-Roformer vocals model -> instrumental -> htdemucs_ft -> other
    bsrof_ft   BS-Roformer-SW harmonic mix (guitar + piano + other at native rate) -> htdemucs_ft -> other
               (removes the vocal / drum / bass bleed that is left in the bsrof_sw harmonic mix)
    bsrof_bass      test: bsrof_sw guitar + piano + other + BASS (still no vocals / drums)
    demucs_ft_bass  test: htdemucs_ft other + BASS (still no vocals / drums)

``run`` writes   DE/data/export/pilot/<variant>/<song_id>.wav   (24 kHz mono, harmonic stem)
                 DE/data/export/pilot/speed.csv                 (seconds, audio seconds, peak VRAM per call)
``score`` writes DE/data/export/pilot/scores.csv + summary on stdout + DE/data/export/pilot/index.html
                 (a listening page: mix and every variant side by side for each song)

Metric (chroma against the annotation, no model needed): per song and variant the mean cosine between
the chroma frame and the template of the annotated chord, best shift in +-0.3 s; plus root accuracy
and root+quality accuracy of the frame-wise best template out of 12 x 6 chord templates.
A better separation should raise these relative to ``mix`` on the SAME songs.
"""
from __future__ import annotations
import argparse
import csv
import html
import json
import os
import re
import shutil
import signal
import sys
import time
from pathlib import Path

if "--cpu" in sys.argv:                                     # must be decided before torch is imported
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, "/opt/airflow")

import numpy as np          # noqa: E402
import soundfile as sf      # noqa: E402

ROOT = Path("/data/export/pilot")
STEM_TMP = Path("/data/tmp/pilot_stems")
SR = 24_000
HOP = 2048
HARMONIC = {"guitar", "piano", "other"}                     # stems that are kept in a 6-stem model
VARIANTS = {
    "demucs6s":  ("htdemucs_6s.yaml", HARMONIC),
    "demucs_ft": ("htdemucs_ft.yaml", {"other"}),
    "bsrof_sw":  ("BS-Roformer-SW.ckpt", HARMONIC),
    "chain":     (None, None),
    "bsrof_ft":  (None, None),
    "bsrof_bass":     ("BS-Roformer-SW.ckpt", HARMONIC | {"bass"}),   # test: harmonic + bass (no vocals / drums)
    "demucs_ft_bass": ("htdemucs_ft.yaml", {"other", "bass"}),         # test: other + bass (no vocals / drums)
}
VOCAL_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"


def stem_name(path: str) -> str:
    m = re.findall(r"\(([^)]+)\)", Path(path).name)
    return m[0].lower() if m else ""


def load_mono(path: Path, sr: int) -> np.ndarray:
    import librosa
    y, file_sr = sf.read(str(path), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    return librosa.resample(y, orig_sr=file_sr, target_sr=sr) if file_sr != sr else y


def make_separator(model: str, out_dir: Path, demucs_segment: int):
    from audio_separator.separator import Separator
    sep = Separator(
        output_dir=str(out_dir), output_format="WAV", log_level=30,
        model_file_dir="/home/airflow/.cache/audio-separator-models/",
        demucs_params={"segment_size": demucs_segment, "shifts": 1, "overlap": 0.25, "segments_enabled": True},
        mdxc_params={"segment_size": 256, "override_model_segment_size": False, "batch_size": 1,
                     "overlap": 4, "pitch_shift": 0},
    )
    sep.load_model(model_filename=model)
    return sep


def mix_stems(paths: list[Path], dest: Path) -> Path:
    """Sum stems at their native rate (peak capped at 0.99) -> one wav, to feed a second separator."""
    mix, sr = None, None
    for p in paths:
        y, file_sr = sf.read(str(p), dtype="float32", always_2d=True)
        if mix is None:
            mix, sr = y, file_sr
        else:
            n = min(len(mix), len(y))
            mix = mix[:n] + y[:n]
    if mix is None:
        raise RuntimeError("no harmonic stems to mix")
    peak = float(np.abs(mix).max())
    if peak > 0.99:
        mix = mix * (0.99 / peak)
    sf.write(str(dest), mix, sr)
    return dest


def run_model(sep, wav: Path, out_dir: Path) -> dict[str, Path]:
    files = sep.separate(str(wav))
    return {stem_name(f): out_dir / Path(f).name for f in files}


# --------------------------------------------------------------------------------------------- run
def cmd_run(args) -> None:
    import fcntl
    import torch
    lock = open("/data/tmp/pilot_run.lock", "w")          # one run at a time: parallel runs fight over the GPU
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit("another pilot_separate run is already active - refusing to start a second one")

    def on_timeout(signum, frame):
        raise TimeoutError(f"separation exceeded {args.timeout}s")
    signal.signal(signal.SIGALRM, on_timeout)
    manifest = json.loads((ROOT / "manifest.json").read_text())
    if args.limit:
        manifest = manifest[: args.limit]
    speed_path = ROOT / "speed.csv"
    new = not speed_path.exists()
    speed = open(speed_path, "a", newline="")
    sw = csv.writer(speed)
    if new:
        sw.writerow(["variant", "song_id", "audio_s", "seconds", "rtf", "peak_vram_mib", "device", "error"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for variant in args.variants:
        out_variant = ROOT / variant
        out_variant.mkdir(parents=True, exist_ok=True)
        tmp = STEM_TMP / f"{variant}_{os.getpid()}"       # per-process: never share stems between runs
        tmp.mkdir(parents=True, exist_ok=True)
        model, keep = VARIANTS[variant]
        if variant == "chain":
            seps = [make_separator(VOCAL_MODEL, tmp, args.demucs_segment),
                    make_separator("htdemucs_ft.yaml", tmp, args.demucs_segment)]
        elif variant == "bsrof_ft":
            seps = [make_separator(VARIANTS["bsrof_sw"][0], tmp, args.demucs_segment),
                    make_separator("htdemucs_ft.yaml", tmp, args.demucs_segment)]
        else:
            seps = [make_separator(model, tmp, args.demucs_segment)]
        print(f"== {variant} on {device}")
        for i, rec in enumerate(manifest, 1):
            sid = rec["song_id"]
            dest = out_variant / f"{sid}.wav"
            if dest.exists():
                continue
            raw = ROOT / "raw" / f"{sid}.wav"
            error = ""
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            signal.alarm(args.timeout)
            try:
                if variant == "chain":
                    first = run_model(seps[0], raw, tmp)
                    inst = next(p for n, p in first.items() if n in ("instrumental", "other"))
                    second = run_model(seps[1], inst, tmp)
                    parts = [p for n, p in second.items() if n == "other"]
                elif variant == "bsrof_ft":
                    first = run_model(seps[0], raw, tmp)
                    harm = mix_stems([p for n, p in first.items() if n in HARMONIC], tmp / "harm_mix.wav")
                    second = run_model(seps[1], harm, tmp)
                    parts = [p for n, p in second.items() if n == "other"]
                else:
                    stems = run_model(seps[0], raw, tmp)
                    parts = [p for n, p in stems.items() if n in keep]
                dt = time.perf_counter() - t0
                mixed = None
                for p in parts:
                    y = load_mono(p, SR)
                    mixed = y if mixed is None else mixed[: len(y)] + y[: len(mixed)]
                peak = float(np.abs(mixed).max())
                if peak > 0.99:                                     # summed stems (esp. with bass) may exceed full scale
                    mixed = mixed * (0.99 / peak)
                sf.write(str(dest), mixed, SR, subtype="PCM_16")
            except Exception as e:          # OOM or a broken file must not stop the batch
                dt = time.perf_counter() - t0
                error = f"{type(e).__name__}: {str(e)[:120]}"
                print(f"  [{i}] {sid[:8]} ERROR {error}")
            finally:
                signal.alarm(0)
                for f in tmp.glob("*"):
                    f.unlink(missing_ok=True)
                if device == "cuda":
                    torch.cuda.empty_cache()
            peak = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0
            sw.writerow([variant, sid, rec["duration_s"], round(dt, 2), round(dt / rec["duration_s"], 2),
                         round(peak), device, error])
            speed.flush()
            if i % 5 == 0:
                print(f"  {i}/{len(manifest)}  last {dt:.1f}s for {rec['duration_s']}s audio, peak {peak:.0f} MiB")
        shutil.rmtree(tmp, ignore_errors=True)
        del seps
        if device == "cuda":
            torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------------- score
TRIADS = {0: (0, 4, 7), 1: (0, 3, 7), 2: (0, 4, 7, 10), 3: (0, 3, 6), 4: (0, 2, 7), 5: (0, 5, 7)}
SEVENTH = {3: 11, 4: 10, 5: 10}


def template(root: int, quality: int, ext: int | None) -> np.ndarray:
    pcs = set(TRIADS[quality])
    if ext in SEVENTH and quality != 2:
        pcs.add(9 if (quality == 3 and ext == 5) else SEVENTH[ext])
    v = np.zeros(12, dtype=np.float32)
    for p in pcs:
        v[(root + p) % 12] = 1.0
    return v / np.linalg.norm(v)


def cmd_score(args) -> None:
    import librosa
    from src.data_processing.application.annotation_parsers.parsers import get_parser
    from src.data_processing.application.mappers.harte_mapper import map_harte

    manifest = json.loads((ROOT / "manifest.json").read_text())
    bank, bank_cls = [], []                                    # 72 plain chord templates
    for r in range(12):
        for q in range(6):
            bank.append(template(r, q, None))
            bank_cls.append((r, q))
    bank = np.stack(bank)
    shifts = np.round(np.arange(-0.3, 0.3001, 0.1), 2)

    rows, songs = [], {}
    for rec in manifest:
        sid, off = rec["song_id"], rec["offset_s"]
        ann = Path(rec["annotation"])
        timeline = []
        for s, e, label in get_parser(ann).parse(ann):
            m = map_harte(label)
            if m.root_id is not None and m.quality_id is not None:
                timeline.append((s, e, m.root_id, m.quality_id, template(m.root_id, m.quality_id, m.ext_id), label))
        if not timeline:
            continue
        starts = np.array([t[0] for t in timeline])
        songs[sid] = {"rec": rec, "chords": [(round(t[0] - off, 1), t[5]) for t in timeline
                                              if 0 <= t[0] - off <= rec["duration_s"]]}
        for variant in ["mix"] + args.variants:
            path = ROOT / "raw" / f"{sid}.wav" if variant == "mix" else ROOT / variant / f"{sid}.wav"
            if not path.exists():
                continue
            y = load_mono(path, SR)
            chroma = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP)
            norm = np.linalg.norm(chroma, axis=0)
            chroma = chroma / np.maximum(norm, 1e-6)
            t = np.arange(chroma.shape[1]) * HOP / SR + off
            best = None
            for s in shifts:
                idx = np.searchsorted(starts, t + s, side="right") - 1
                ok = (idx >= 0) & (norm > 1e-3)
                ok &= (t + s) < np.array([timeline[min(i, len(timeline) - 1)][1] for i in idx])
                if ok.sum() < 20:
                    continue
                cos = np.array([timeline[i][4] @ chroma[:, j] for j, i in zip(np.nonzero(ok)[0], idx[ok])])
                if best is None or cos.mean() > best[0]:
                    best = (float(cos.mean()), s, ok, idx)
            if best is None:
                continue
            cos_mean, s, ok, idx = best
            pred = (bank @ chroma[:, ok]).argmax(axis=0)
            truth = [(timeline[i][2], timeline[i][3]) for i in idx[ok]]
            root_acc = np.mean([bank_cls[p][0] == r for p, (r, _) in zip(pred, truth)])
            triad_acc = np.mean([bank_cls[p] == tq for p, tq in zip(pred, truth)])
            rows.append({"song_id": sid, "source": rec["source"], "variant": variant, "cos": cos_mean,
                         "root_acc": root_acc, "triad_acc": triad_acc, "best_shift": s})

    with open(ROOT / "scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{'variant':<10}{'source':<11}{'n':>4}{'cos':>8}{'root_acc':>10}{'triad_acc':>11}")
    for source in ("hooktheory", "jaah", "all"):
        for variant in ["mix"] + args.variants:
            sel = [r for r in rows if r["variant"] == variant and (source == "all" or r["source"] == source)]
            if sel:
                print(f"{variant:<10}{source:<11}{len(sel):>4}{np.mean([r['cos'] for r in sel]):>8.3f}"
                      f"{np.mean([r['root_acc'] for r in sel]):>10.3f}{np.mean([r['triad_acc'] for r in sel]):>11.3f}")
    write_listen_page(songs, args.variants)
    print("listening page:", ROOT / "index.html")


def write_listen_page(songs: dict, variants: list[str]) -> None:
    head = ("<!doctype html><meta charset=utf-8><title>Separator pilot</title><style>"
            "body{font:14px sans-serif;background:#111;color:#ddd;margin:20px}table{border-collapse:collapse}"
            "td,th{border:1px solid #333;padding:6px;vertical-align:top}audio{width:230px;height:32px}"
            ".ch{max-width:330px;font-size:12px;color:#9ab}</style><h2>Separator pilot: raw mix vs separated stems</h2>"
            "<table><tr><th>song</th><th>mix</th>" + "".join(f"<th>{v}</th>" for v in variants) + "<th>chords (s)</th></tr>")
    body = []
    for sid, d in songs.items():
        cells = [f"<td>{d['rec']['source']}<br>{sid[:8]}</td>", f"<td><audio controls preload=none src='raw/{sid}.wav'></audio></td>"]
        for v in variants:
            ok = (ROOT / v / f"{sid}.wav").exists()
            cells.append(f"<td><audio controls preload=none src='{v}/{sid}.wav'></audio></td>" if ok else "<td>-</td>")
        chords = " | ".join(f"{t}s {html.escape(c)}" for t, c in d["chords"][:14])
        cells.append(f"<td class=ch>{chords}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    (ROOT / "index.html").write_text(head + "".join(body) + "</table>", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "score"):
        s = sub.add_parser(name)
        s.add_argument("--variants", nargs="+", default=["demucs6s", "demucs_ft", "bsrof_sw", "chain"],
                       choices=list(VARIANTS))
        s.add_argument("--cpu", action="store_true", help="hide the GPU (use when VRAM is full)")
        if name == "run":
            s.add_argument("--limit", type=int, default=0)
            s.add_argument("--demucs-segment", type=int, default=6, help="Demucs chunk seconds (VRAM knob)")
            s.add_argument("--timeout", type=int, default=240, help="give up on one song after this many seconds")
    args = p.parse_args()
    (cmd_run if args.cmd == "run" else cmd_score)(args)


if __name__ == "__main__":
    main()
