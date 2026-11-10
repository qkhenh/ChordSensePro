from __future__ import annotations

from collections import Counter
import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

# Add PyTorch bundled CUDA DLLs to PATH so onnxruntime-gpu can find them
try:
    import torch as _torch
    _torch_lib = os.path.join(os.path.dirname(_torch.__file__), "lib")
    os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")
except ImportError:
    pass

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


CHORD_SUFFIX: dict[tuple[int, int], str] = {
    # major (0)
    (0, 0): "", (0, 1): "b5", (0, 2): "aug", (0, 3): "maj7", (0, 6): "add9",
    (0, 8): "maj9", (0, 10): "maj11", (0, 11): "maj7#11", (0, 12): "maj13",
    # minor (1)
    (1, 0): "m", (1, 3): "m(maj7)", (1, 4): "m7", (1, 6): "madd9", (1, 7): "m7b9",
    (1, 8): "m9", (1, 10): "m11", (1, 11): "m7#11", (1, 12): "m13", (1, 13): "m7b13",
    # dominant (2)
    (2, 1): "7b5", (2, 2): "7#5", (2, 5): "7", (2, 7): "7b9", (2, 8): "9",
    (2, 9): "7#9", (2, 10): "11", (2, 11): "7#11", (2, 12): "13", (2, 13): "7b13", (2, 14): "7alt",
    # diminished (3)
    (3, 0): "dim", (3, 4): "m7b5", (3, 5): "dim7",
    # sus2 (4)
    (4, 0): "sus2", (4, 3): "maj7sus2", (4, 5): "7sus2",
    # sus4 (5)
    (5, 0): "sus4", (5, 3): "maj7sus4", (5, 5): "7sus4", (5, 8): "9sus4", (5, 12): "13sus4",
}


def chord_name(root_id: int, quality_id: int, ext_id: int, bass_id: int = 12) -> str:
    if root_id < 0:
        return "N"
    r = ROOT_NAMES[root_id]
    suffix = CHORD_SUFFIX.get((quality_id, ext_id))
    if suffix is None:
        q = QUALITY_NAMES[quality_id]
        e = EXT_NAMES[ext_id]
        suffix = QUALITY_SUFFIX[q] + EXT_SUFFIX[e]
    name = r + suffix
    if 0 <= bass_id < 12 and bass_id != root_id:
        name += "/" + ROOT_NAMES[bass_id]
    return name


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
        candidates = [classes_json] if classes_json else []
        candidates.extend([
            "../DE/data/export/v3/classes.json",
            "../DE/data/export/v2/classes.json",
            "DE/data/export/v3/classes.json",
            "DE/data/export/v2/classes.json",
        ])
        for cand in candidates:
            if cand and Path(cand).exists():
                data = json.loads(Path(cand).read_text(encoding="utf-8"))
                if "valid_combo_mask" in data:
                    self.combo_mask = np.array(data["valid_combo_mask"], dtype=bool)
                    log.info(f"Loaded valid_combo_mask from {cand}")
                    break

    def predict(self, wav_float32: np.ndarray) -> tuple[int, int, int, int, float]:
        inp = wav_float32[np.newaxis, :]
        outputs = self.session.run(None, {self.input_name: inp})
        r_logits, q_logits, e_logits = outputs[0], outputs[1], outputs[2]

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

        if len(outputs) >= 4:
            b_logits = outputs[3]
            bass_prob = _softmax(b_logits[0])
            bass_id = int(bass_prob.argmax())
            bass_conf = float(bass_prob[bass_id])
            avg_conf = (root_conf + qual_conf + ext_conf + bass_conf) / 4
        else:
            bass_id = 12
            avg_conf = (root_conf + qual_conf + ext_conf) / 3

        return root_id, qual_id, ext_id, bass_id, avg_conf


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


def load_audio_bytes(data: bytes, target_sr: int, filename: str = "") -> np.ndarray:
    import importlib
    ext = Path(filename).suffix.lower() if filename else ""

    if ext in ("", ".wav"):
        try:
            import scipy.io.wavfile as wavfile
            buf = io.BytesIO(data)
            sr, wav = wavfile.read(buf)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            wav = wav.astype(np.float32)
            if wav.max() > 1.0:
                wav = wav / 32768.0
            if sr != target_sr:
                ta = importlib.import_module("torchaudio")
                torch = importlib.import_module("torch")
                wav_t = torch.from_numpy(wav).unsqueeze(0)
                wav_t = ta.functional.resample(wav_t, sr, target_sr)
                wav = wav_t.squeeze(0).numpy()
            return (wav * 32767).clip(-32768, 32767).astype(np.int16)
        except Exception as exc:
            log.debug(f"scipy WAV decode failed: {exc}, falling back to torchaudio")

    try:
        import librosa
        wav, sr_orig = librosa.load(io.BytesIO(data), sr=None, mono=True)
        if sr_orig != target_sr:
            wav = librosa.resample(wav, orig_sr=sr_orig, target_sr=target_sr)
        return (wav * 32767).clip(-32768, 32767).astype(np.int16)
    except ImportError:
        raise RuntimeError("librosa not installed: pip install librosa")


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

    raw_preds: list[dict[str, Any]] = []
    for pos, crop in windows:
        silent = not _rms_gate(crop, rms_threshold_db)
        if silent:
            raw_preds.append({"pos": pos, "root_id": -1, "quality_id": -1, "ext_id": -1, "bass_id": 12, "conf": 0.0, "silent": True})
            continue
        wav = _normalize(crop)
        root_id, qual_id, ext_id, bass_id, conf = predictor.predict(wav)
        reliable = conf >= confidence_threshold
        if not reliable:
            root_id, qual_id, ext_id, bass_id = -1, -1, -1, 12
        raw_preds.append({"pos": pos, "root_id": root_id, "quality_id": qual_id, "ext_id": ext_id, "bass_id": bass_id, "conf": conf, "silent": False})

    return _vote_and_merge(raw_preds, audio_len, sample_rate, step_samples, crop_samples)


def _vote_and_merge(
    raw_preds: list[dict],
    audio_len: int,
    sample_rate: int,
    step_samples: int,
    crop_samples: int,
) -> list[dict[str, Any]]:
    if not raw_preds:
        return []

    total_frames = max(p["pos"] for p in raw_preds) // step_samples + 1
    votes: list[list[tuple[int, int, int, int, float, bool]]] = [[] for _ in range(total_frames)]

    for pred in raw_preds:
        frame_start = pred["pos"] // step_samples
        frames_covered = (crop_samples + step_samples - 1) // step_samples
        b_id = pred.get("bass_id", 12)
        for fi in range(frame_start, min(frame_start + frames_covered, total_frames)):
            votes[fi].append((pred["root_id"], pred["quality_id"], pred["ext_id"], b_id, pred["conf"], pred["silent"]))

    frame_results: list[dict[str, Any]] = []
    for fi, frame_votes in enumerate(votes):
        if not frame_votes:
            continue
        pos = fi * step_samples
        start_ms = round(pos / sample_rate * 1000, 1)
        end_ms = round((pos + step_samples) / sample_rate * 1000, 1)

        silent_count = sum(1 for v in frame_votes if v[5])
        if silent_count > len(frame_votes) / 2:
            frame_results.append({"start_ms": start_ms, "end_ms": end_ms, "root_id": -1, "quality_id": -1, "ext_id": -1, "bass_id": 12, "chord": "N", "confidence": 0.0, "silent": True})
            continue

        active = [(r, q, e, b, c) for r, q, e, b, c, s in frame_votes if not s]
        combo_counts: Counter = Counter((r, q, e, b) for r, q, e, b, c in active)
        best_combo, _ = combo_counts.most_common(1)[0]
        best_conf = max(c for r, q, e, b, c in active if (r, q, e, b) == best_combo)
        root_id, qual_id, ext_id, bass_id = best_combo

        frame_results.append({
            "start_ms": start_ms,
            "end_ms": end_ms,
            "root_id": root_id,
            "quality_id": qual_id,
            "ext_id": ext_id,
            "bass_id": bass_id,
            "chord": chord_name(root_id, qual_id, ext_id, bass_id),
            "confidence": round(best_conf, 4),
            "silent": False,
        })

    smoothed = _smooth_jitter(frame_results, max_jitter_frames=1)
    return _merge_consecutive(smoothed)


def _chord_key(r: dict) -> tuple[int, int, int, int]:
    return (
        r["root_id"],
        r["quality_id"],
        r["ext_id"],
        r.get("bass_id", 12),
    )


def _smooth_jitter(results: list[dict], max_jitter_frames: int = 1) -> list[dict]:
    if len(results) < 3 or max_jitter_frames < 1:
        return [r.copy() for r in results]

    out = [r.copy() for r in results]
    n = len(out)
    i = 1
    while i < n - 1:
        run_end = i
        while run_end + 1 < n and _chord_key(out[run_end + 1]) == _chord_key(out[i]):
            run_end += 1
        run_len = run_end - i + 1

        if (
            run_len <= max_jitter_frames
            and run_end + 1 < n
            and _chord_key(out[i - 1]) == _chord_key(out[run_end + 1])
            and _chord_key(out[i]) != _chord_key(out[i - 1])
        ):
            ref = out[i - 1]
            for k in range(i, run_end + 1):
                out[k]["root_id"] = ref["root_id"]
                out[k]["quality_id"] = ref["quality_id"]
                out[k]["ext_id"] = ref["ext_id"]
                out[k]["bass_id"] = ref.get("bass_id", 12)
                out[k]["chord"] = ref["chord"]
                out[k]["silent"] = ref["silent"]
        i = run_end + 1

    return out


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
            and r.get("bass_id", 12) == prev.get("bass_id", 12)
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
            audio = load_audio_bytes(data, SAMPLE_RATE, file.filename or "")
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

    global _predictor
    _model_path = args.model

    if not Path(_model_path).exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    log.info(f"Loading ONNX model: {_model_path}")
    _predictor = ONNXPredictor(_model_path, args.classes_json)
    log.info("Model loaded. Starting server...")

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)


if __name__ == "__main__":
    main()

