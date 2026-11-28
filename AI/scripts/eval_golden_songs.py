from __future__ import annotations

"""Evaluate any v5 or v6 ONNX model against reviewed Golden Songs & Phép Màu Intro/Band/Outro probe."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.serve import HarmonicSeparator, ONNXPredictor, infer_audio, load_audio_bytes


def _norm_chord(c: str) -> str:
    return c.strip().replace(" ", "")


def eval_reviewed_json(
    predictor: ONNXPredictor,
    json_path: Path,
    audio_path: Path | None,
    separator: HarmonicSeparator | None,
) -> dict[str, float]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if audio_path is None or not audio_path.exists():
        # Offline score directly from reviewed JSON entries if audio isn't provided
        rows = data if isinstance(data, list) else data.get("chords", data.get("results", []))
        total = len(rows)
        ok = sum(1 for r in rows if r.get("is_correct") or r.get("correct"))
        return {"total": total, "correct": ok, "acc": ok / max(total, 1)}

    raw_bytes = audio_path.read_bytes()
    wav = load_audio_bytes(
        raw_bytes,
        target_sr=24000,
        filename=audio_path.name,
        separate=(separator is not None),
        separator=separator,
        demucs_mode="auto",
    )
    preds = infer_audio(predictor, wav, sample_rate=24000)

    gt_rows = data if isinstance(data, list) else data.get("chords", data.get("results", []))
    matched = 0
    total = 0
    short_ok = 0
    short_tot = 0

    for row in gt_rows:
        s_ms = float(row.get("start_ms", row.get("start", 0.0) * 1000.0))
        e_ms = float(row.get("end_ms", row.get("end", 0.0) * 1000.0))
        gt_chord = row.get("expected") or row.get("ground_truth") or (row.get("chord") if row.get("is_correct") else None)
        if not gt_chord or e_ms <= s_ms:
            continue
        mid = 0.5 * (s_ms + e_ms)
        pred_c = "N"
        for p in preds:
            if float(p["start_ms"]) <= mid <= float(p["end_ms"]):
                pred_c = str(p["chord"])
                break
        is_hit = _norm_chord(pred_c) == _norm_chord(str(gt_chord))
        total += 1
        matched += int(is_hit)
        if (e_ms - s_ms) <= 800.0:
            short_tot += 1
            short_ok += int(is_hit)

    return {
        "total": total,
        "correct": matched,
        "acc": matched / max(total, 1),
        "short_total": short_tot,
        "short_acc": short_ok / max(short_tot, 1) if short_tot else 0.0,
        "n_pred_chords": len(preds),
    }


def probe_phep_mau(
    predictor: ONNXPredictor,
    audio_path: Path,
    separator: HarmonicSeparator | None,
) -> None:
    wav = load_audio_bytes(
        audio_path.read_bytes(),
        target_sr=24000,
        filename=audio_path.name,
        separate=(separator is not None),
        separator=separator,
        demucs_mode="auto",
    )
    preds = infer_audio(predictor, wav, sample_rate=24000)
    print(f"\n=== Phép Màu Probe ({len(preds)} chords) ===")
    for label, t0, t1 in [("Intro Solo Piano (0-55s)", 0, 55000), ("Full Band (55-180s)", 55000, 180000), ("Outro Piano (200-257s)", 200000, 260000)]:
        seg_chords = [
            f"{p['start_ms']/1000:.1f}s:{p['chord']}({p['confidence']:.2f})"
            for p in preds
            if t0 <= float(p["start_ms"]) < t1
        ]
        print(f"  {label}: {' | '.join(seg_chords[:18])}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate ONNX model on Golden Songs & Phép Màu")
    ap.add_argument("--model", required=True, help="Path to ONNX encoder model")
    ap.add_argument("--no-demucs", action="store_true")
    ap.add_argument("--phep-mau", default="songs/Phép Màu (Đàn Cá Gỗ OST) - MAYDAYs ft. Minh Tốc  Official Lyric Video - MAYDAYs.mp3")
    ap.add_argument("--json", nargs="*", default=[])
    ap.add_argument("--audio", nargs="*", default=[])
    args = ap.parse_args()

    predictor = ONNXPredictor(args.model)
    sep = None if args.no_demucs else HarmonicSeparator()

    for idx, j_str in enumerate(args.json):
        jp = Path(j_str)
        ap_path = Path(args.audio[idx]) if idx < len(args.audio) else None
        if jp.exists():
            res = eval_reviewed_json(predictor, jp, ap_path, sep)
            print(f"[Golden] {jp.name}: {res}")

    pm = Path(args.phep_mau)
    if pm.exists():
        probe_phep_mau(predictor, pm, sep)


if __name__ == "__main__":
    main()
