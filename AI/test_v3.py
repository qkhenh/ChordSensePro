from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.serve import HarmonicSeparator, ONNXPredictor, infer_audio, load_audio_bytes

DEFAULT_MODELS = [
    "results/model_v4_5s_p2.onnx",
    "results/model_v3_5s_p2.onnx",
]


def resolve_default_model() -> str:
    for m in DEFAULT_MODELS:
        if os.path.exists(m):
            return m
    return DEFAULT_MODELS[0]


def resolve_default_audio() -> str:
    for p in (r"D:\full.mp3", r"D:\test.wav"):
        if os.path.exists(p):
            return p
    return r"D:\full.mp3"


def main() -> None:
    ap = argparse.ArgumentParser(description="Real-song inference with Demucs htdemucs_ft (other + bass) + ONNX")
    ap.add_argument("--audio", default=resolve_default_audio(), help="Path to mp3/wav/flac audio file")
    ap.add_argument("--model", default=resolve_default_model(), help="Path to exported .onnx model")
    ap.add_argument("--classes-json", default="../DE/data/export/v3/classes.json")
    ap.add_argument("--step-s", type=float, default=2.5)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--no-demucs", action="store_true", help="Skip htdemucs_ft stem separation (only for pre-separated stems)")
    args = ap.parse_args()

    if not os.path.exists(args.audio):
        print("Audio file not found:", args.audio)
        sys.exit(1)
    if not os.path.exists(args.model):
        print("ONNX model not found:", args.model)
        sys.exit(1)

    print(f"Loading ONNX model: {args.model}")
    pred = ONNXPredictor(args.model, args.classes_json)
    print(f"Mode: {'frame-level (~75 Hz)' if pred.is_frame_level else 'clip-level'}")

    sep = None
    if not args.no_demucs:
        print("Loading Demucs model: htdemucs_ft (keeping stems: other + bass) ...")
        sep = HarmonicSeparator(model_name="htdemucs_ft", segment=6.0)

    print(f"Loading audio: {args.audio}")
    t0 = time.perf_counter()
    with open(args.audio, "rb") as f:
        raw_bytes = f.read()
    audio = load_audio_bytes(
        raw_bytes,
        target_sr=24000,
        filename=args.audio,
        separate=(sep is not None),
        separator=sep,
    )
    prep_s = time.perf_counter() - t0
    dur_s = len(audio) / 24000.0
    print(f"Audio duration: {dur_s:.1f}s | Prep + Demucs: {prep_s:.2f}s")

    t1 = time.perf_counter()
    results = infer_audio(pred, audio, step_s=args.step_s, confidence_threshold=args.conf)
    infer_s = time.perf_counter() - t1

    print(f"\n{'Start':>7}  {'End':>7}  {'Chord':<14} {'Conf':>5}")
    print("-" * 42)
    for r in results:
        start = r["start_ms"] / 1000.0
        end = r["end_ms"] / 1000.0
        print(f"{start:6.2f}s  {end:6.2f}s  {r['chord']:<14} {r['confidence']:.2f}")

    print(f"\nTotal segments: {len(results)} | ONNX inference: {infer_s:.2f}s")


if __name__ == "__main__":
    main()
