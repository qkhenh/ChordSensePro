from __future__ import annotations

import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False

try:
    import torchaudio
    _TORCHAUDIO_AVAILABLE = True
except ImportError:
    _TORCHAUDIO_AVAILABLE = False

try:
    from fastapi import FastAPI, File, HTTPException, Query, UploadFile
    from fastapi.responses import JSONResponse
    import uvicorn
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False


SAMPLE_RATE = int(os.environ.get("SERVE_SAMPLE_RATE", "24000"))
CROP_S = int(os.environ.get("SERVE_CROP_S", "5"))
STEP_S = float(os.environ.get("SERVE_STEP_S", "1.0"))
RMS_THRESHOLD_DB = float(os.environ.get("SERVE_RMS_THRESHOLD_DB", "-40.0"))
CONFIDENCE_THRESHOLD = float(os.environ.get("SERVE_CONFIDENCE_THRESHOLD", "0.3"))
HOST = os.environ.get("SERVE_HOST", "0.0.0.0")
PORT = int(os.environ.get("SERVE_PORT", "8000"))
MODEL_PATH = os.environ.get("SERVE_MODEL_PATH", "results/model_p2.onnx")
CLASSES_JSON = os.environ.get("SERVE_CLASSES_JSON", "../DE/data/export/v2/classes.json")

ROOT_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
QUALITY_NAMES = ["maj", "min", "dom", "dim", "sus2", "sus4"]
EXT_NAMES = ["", "b5", "#5", "maj7", "m7", "7", "add9", "b9", "9", "#9", "11", "#11", "13", "b13", "alt"]

QUALITY_SUFFIX = {
    "maj": "",
    "min": "m",
    "dom": "",
    "dim": "dim",
    "sus2": "sus2",
    "sus4": "sus4",
}
EXT_SUFFIX = {
    "": "",
    "b5": "(b5)",
    "#5": "#5",
    "maj7": "maj7",
    "m7": "m7",
    "7": "7",
    "add9": "add9",
    "b9": "b9",
    "9": "9",
    "#9": "#9",
    "11": "11",
    "#11": "#11",
    "13": "13",
    "b13": "b13",
    "alt": "alt",
}


def chord_name(root_id: int, quality_id: int, ext_id: int) -> str:
    if root_id < 0:
        return "N"
    r = ROOT_NAMES[root_id]
    q = QUALITY_NAMES[quality_id]
    e = EXT_NAMES[ext_id]
    return r + QUALITY_SUFFIX[q] + EXT_SUFFIX[e]


class ONNXPredictor:
    def __init__(self, model_path: str, classes_json: str | None = None):
        if not _ORT_AVAILABLE:
            raise RuntimeError("onnxruntime not installed: pip install onnxruntime-gpu")

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)

        active = self.session.get_providers()[0]
        log.info(f"ONNX model loaded: {model_path} | provider: {active}")

        self.input_name = self.session.get_inputs()[0].name

        self.combo_mask: np.ndarray | None = None
        if classes_json and Path(classes_json).exists():
            data = json.loads(Path(classes_json).read_text(encoding="utf-8"))
            if "valid_combo_mask" in data:
                self.combo_mask = np.array(data["valid_combo_mask"], dtype=bool)
                log.info(f"Loaded valid_combo_mask from {classes_json}")

    def predict(self, wav_float32: np.ndarray) -> tuple[int, int, int, float]:
        inp = wav_float32[np.newaxis, :]
        r_logits, q_logits, e_logits = self.session.run(None, {self.input_name: inp})

        root_prob = _softmax(r_logits[0])
        qual_prob = _softmax(q_logits[0])
        ext_prob = _softmax(e_logits[0])

        root_id = int(root_prob.argmax())
        root_conf = float(root_prob[root_id])

        qual_id = int(qual_prob.argmax())
        qual_conf = float(qual_prob[qual_id])

        if self.combo_mask is not None:
            masked = ext_prob.copy()
            masked[~self.combo_mask[qual_id]] = 0.0
            s = masked.sum()
            if s > 0:
                masked /= s
            ext_id = int(masked.argmax())
            ext_conf = float(masked[ext_id])
        else:
            ext_id = int(ext_prob.argmax())
            ext_conf = float(ext_prob[ext_id])

        avg_conf = (root_conf + qual_conf + ext_conf) / 3
        return root_id, qual_id, ext_id, avg_conf


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _rms_gate(crop: np.ndarray, threshold_db: float) -> bool:
    rms = float(np.sqrt(np.mean(crop.astype(np.float32) ** 2) + 1e-10))
    rms_db = 20 * np.log10(rms / 32768.0 + 1e-10)
    return rms_db >= threshold_db


def _normalize(wav: np.ndarray) -> np.ndarray:
    wav = wav.astype(np.float32) / 32768.0
    std = wav.std()
    if std > 1e-6:
        wav = (wav - wav.mean()) / std
    return wav


def load_audio_bytes(data: bytes, target_sr: int) -> np.ndarray:
    if not _TORCHAUDIO_AVAILABLE:
        raise RuntimeError("torchaudio not installed")
    import torch
    buf = io.BytesIO(data)
    wav, sr = torchaudio.load(buf)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    arr = (wav.squeeze(0).numpy() * 32767).astype(np.int16)
    return arr


def infer_audio(
    predictor: ONNXPredictor,
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    crop_s: int = CROP_S,
    step_s: float = STEP_S,
    rms_threshold_db: float = RMS_THRESHOLD_DB,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> list[dict[str, Any]]:
    crop_samples = crop_s * sample_rate
    step_samples = int(step_s * sample_rate)
    audio_len = len(audio)

    windows: list[tuple[int, np.ndarray]] = []
    if audio_len < crop_samples:
        pad = np.zeros(crop_samples, dtype=np.int16)
        pad[:audio_len] = audio
        windows.append((0, pad))
    else:
        pos = 0
        while pos + crop_samples <= audio_len:
            windows.append((pos, audio[pos: pos + crop_samples]))
            pos += step_samples
        if pos < audio_len:
            last = audio_len - crop_samples
            if last != pos - step_samples:
                windows.append((last, audio[last: last + crop_samples]))

    results = []
    for pos, crop in windows:
        start_ms = round(pos / sample_rate * 1000, 1)
        end_ms = round((pos + crop_samples) / sample_rate * 1000, 1)

        if not _rms_gate(crop, rms_threshold_db):
            results.append({
                "start_ms": start_ms,
                "end_ms": end_ms,
                "root_id": -1,
                "quality_id": -1,
                "ext_id": -1,
                "chord": "N",
                "confidence": 0.0,
                "reliable": False,
                "silent": True,
            })
            continue

        wav = _normalize(crop)
        t0 = time.perf_counter()
        root_id, qual_id, ext_id, conf = predictor.predict(wav)
        infer_ms = round((time.perf_counter() - t0) * 1000, 1)

        reliable = conf >= confidence_threshold
        if not reliable:
            root_id, qual_id, ext_id = -1, -1, -1

        results.append({
            "start_ms": start_ms,
            "end_ms": end_ms,
            "root_id": root_id,
            "quality_id": qual_id,
            "ext_id": ext_id,
            "chord": chord_name(root_id, qual_id, ext_id),
            "confidence": round(conf, 4),
            "reliable": reliable,
            "silent": False,
            "_infer_ms": infer_ms,
        })

    return _merge_consecutive(results)


def _merge_consecutive(results: list[dict]) -> list[dict]:
    if not results:
        return results
    merged = [results[0].copy()]
    for r in results[1:]:
        prev = merged[-1]
        same = (
            r["root_id"] == prev["root_id"]
            and r["quality_id"] == prev["quality_id"]
            and r["ext_id"] == prev["ext_id"]
        )
        if same:
            prev["end_ms"] = r["end_ms"]
            prev["confidence"] = max(prev["confidence"], r["confidence"])
        else:
            merged.append(r.copy())
    return merged


_predictor: ONNXPredictor | None = None


def get_predictor() -> ONNXPredictor:
    global _predictor
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _predictor


def create_app() -> "FastAPI":
    if not _FASTAPI_AVAILABLE:
        raise RuntimeError("fastapi not installed: pip install fastapi uvicorn")

    app = FastAPI(title="ChordSense Pro — Chord Recognition API", version="1.0.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": MODEL_PATH}

    @app.post("/predict")
    async def predict(
        file: UploadFile = File(...),
        step_s: float = Query(default=STEP_S, ge=0.1, le=5.0),
        rms_threshold_db: float = Query(default=RMS_THRESHOLD_DB),
        confidence_threshold: float = Query(default=CONFIDENCE_THRESHOLD, ge=0.0, le=1.0),
    ):
        if not file.filename.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
            raise HTTPException(status_code=400, detail="Unsupported file format")

        data = await file.read()
        try:
            audio = load_audio_bytes(data, SAMPLE_RATE)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Audio decode failed: {exc}")

        t_start = time.perf_counter()
        predictor = get_predictor()
        segments = infer_audio(
            predictor,
            audio,
            sample_rate=SAMPLE_RATE,
            step_s=step_s,
            rms_threshold_db=rms_threshold_db,
            confidence_threshold=confidence_threshold,
        )
        elapsed_ms = round((time.perf_counter() - t_start) * 1000, 1)

        return JSONResponse({
            "duration_ms": round(len(audio) / SAMPLE_RATE * 1000, 1),
            "segments": len(segments),
            "elapsed_ms": elapsed_ms,
            "chords": segments,
        })

    @app.post("/predict_npy")
    async def predict_npy(
        file: UploadFile = File(...),
        step_s: float = Query(default=STEP_S, ge=0.1, le=5.0),
    ):
        data = await file.read()
        try:
            audio = np.frombuffer(data, dtype=np.int16)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"NPY decode failed: {exc}")

        predictor = get_predictor()
        segments = infer_audio(predictor, audio, sample_rate=SAMPLE_RATE, step_s=step_s)
        return JSONResponse({"segments": len(segments), "chords": segments})

    return app


def main():
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="ChordSense Pro — ONNX Inference Server")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to .onnx model file")
    parser.add_argument("--classes_json", default=CLASSES_JSON)
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--step_s", type=float, default=STEP_S)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    global _predictor, MODEL_PATH
    MODEL_PATH = args.model

    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    log.info(f"Loading ONNX model: {args.model}")
    _predictor = ONNXPredictor(args.model, args.classes_json)
    log.info("Model loaded. Starting server...")

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)


if __name__ == "__main__":
    main()

