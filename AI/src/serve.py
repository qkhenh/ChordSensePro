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


# Exact MERT-v1-330M CNN feature extractor geometry (kept standalone for lean ONNX serving)
MERT_CONV_KERNELS: tuple[int, ...] = (10, 3, 3, 3, 3, 2, 2)
MERT_CONV_STRIDES: tuple[int, ...] = (5, 2, 2, 2, 2, 2, 2)
MERT_HOP_SAMPLES: int = 320
MERT_RECEPTIVE_SAMPLES: int = 400


def mert_output_length(
    input_length: int,
    conv_kernels: tuple[int, ...] = MERT_CONV_KERNELS,
    conv_strides: tuple[int, ...] = MERT_CONV_STRIDES,
) -> int:
    length = int(input_length)
    for k, s in zip(conv_kernels, conv_strides):
        length = (length - k) // s + 1
    return max(0, length)


SAMPLE_RATE = int(os.environ.get("SERVE_SAMPLE_RATE", "24000"))
CROP_S = int(os.environ.get("SERVE_CROP_S", "5"))
STEP_S = float(os.environ.get("SERVE_STEP_S", "2.5"))
RMS_THRESHOLD_DB = float(os.environ.get("SERVE_RMS_THRESHOLD_DB", "-40.0"))
CONFIDENCE_THRESHOLD = float(os.environ.get("SERVE_CONFIDENCE_THRESHOLD", "0.30"))
MIN_DURATION_MS = float(os.environ.get("SERVE_MIN_DURATION_MS", "160.0"))
USE_DEMUCS_DEFAULT = os.environ.get("SERVE_USE_DEMUCS", "1").lower() in ("1", "true", "yes")
DEMUCS_MODEL = os.environ.get("SERVE_DEMUCS_MODEL", "htdemucs_ft")
DEMUCS_SEGMENT = float(os.environ.get("SERVE_DEMUCS_SEGMENT", "6.0"))
HOST = os.environ.get("SERVE_HOST", "0.0.0.0")
PORT = int(os.environ.get("SERVE_PORT", "8000"))
MODEL_PATH = os.environ.get("SERVE_MODEL_PATH", "results/model_v4_5s_p2.onnx")
CLASSES_JSON = os.environ.get("SERVE_CLASSES_JSON", "../DE/data/export/v3/classes.json")

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


def weighted_chord_conf(
    root_conf: float | np.ndarray,
    qual_conf: float | np.ndarray,
    ext_conf: float | np.ndarray,
) -> float | np.ndarray:
    """Compute chord confidence from root/quality/ext only (excluding dominant `bass=none`)."""
    return 0.40 * root_conf + 0.35 * qual_conf + 0.25 * ext_conf


class HarmonicSeparator:
    """Demucs `htdemucs_ft` harmonic stem extractor (`other + bass`, matching DE training v2/v3)."""

    HARMONIC_STEMS = ("other", "bass")

    def __init__(
        self,
        model_name: str = DEMUCS_MODEL,
        segment: float = DEMUCS_SEGMENT,
        device: str | None = None,
    ):
        import torch
        from demucs.pretrained import get_model

        self.model_name = model_name
        self.segment = segment
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_model(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.samplerate = int(getattr(self.model, "samplerate", 44100))
        self.sources = list(getattr(self.model, "sources", ["drums", "bass", "other", "vocals"]))
        log.info(
            f"HarmonicSeparator ready: model={model_name}, sr={self.samplerate}, "
            f"segment={segment}s, device={self.device}"
        )

    def separate_waveform(self, wav_float: np.ndarray, sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
        """Separate `wav_float` (shape `(N,)` or `(C, N)` in `[-1, 1]`) -> mono `int16` at `target_sr`."""
        import torch
        from demucs.apply import apply_model

        if wav_float.ndim == 1:
            stereo = np.stack([wav_float, wav_float], axis=0).astype(np.float32)
        elif wav_float.shape[0] <= 2 and wav_float.shape[0] < wav_float.shape[-1]:
            stereo = wav_float[:2].astype(np.float32)
            if stereo.shape[0] == 1:
                stereo = np.repeat(stereo, 2, axis=0)
        else:
            stereo = wav_float.T[:2].astype(np.float32)
            if stereo.shape[0] == 1:
                stereo = np.repeat(stereo, 2, axis=0)

        wav_t = torch.from_numpy(stereo)
        if sr != self.samplerate:
            if _TORCHAUDIO_AVAILABLE:
                wav_t = torchaudio.functional.resample(wav_t, sr, self.samplerate)
            else:
                import librosa
                ch0 = librosa.resample(stereo[0], orig_sr=sr, target_sr=self.samplerate)
                ch1 = librosa.resample(stereo[1], orig_sr=sr, target_sr=self.samplerate)
                wav_t = torch.from_numpy(np.stack([ch0, ch1], axis=0))

        ref = wav_t.mean(0)
        ref_mean = ref.mean()
        ref_std = ref.std() + 1e-8
        wav_norm = (wav_t - ref_mean) / ref_std

        def _run_apply(dev: str):
            self.model.to(dev)
            with torch.no_grad():
                out = apply_model(
                    self.model,
                    wav_norm.unsqueeze(0).to(dev),
                    segment=self.segment,
                    split=True,
                    overlap=0.25,
                    device=dev,
                )[0]
            return out.cpu() * ref_std + ref_mean

        try:
            stems = _run_apply(self.device)
        except RuntimeError as exc:
            if self.device == "cuda" and "out of memory" in str(exc).lower():
                log.warning("Demucs CUDA OOM — falling back to CPU")
                torch.cuda.empty_cache()
                self.device = "cpu"
                stems = _run_apply("cpu")
            else:
                raise

        mix = None
        for stem_name in self.HARMONIC_STEMS:
            if stem_name in self.sources:
                idx = self.sources.index(stem_name)
                mix = stems[idx] if mix is None else (mix + stems[idx])

        if mix is None:
            raise RuntimeError(f"Demucs model {self.model_name} missing harmonic stems {self.HARMONIC_STEMS}")

        peak = float(mix.abs().max().item()) if mix.numel() else 0.0
        if peak > 0.99:
            mix = mix * (0.99 / peak)

        mono = mix.mean(dim=0)
        if self.samplerate != target_sr:
            if _TORCHAUDIO_AVAILABLE:
                mono = torchaudio.functional.resample(mono.unsqueeze(0), self.samplerate, target_sr).squeeze(0)
                mono_np = mono.numpy()
            else:
                import librosa
                mono_np = librosa.resample(mono.numpy(), orig_sr=self.samplerate, target_sr=target_sr)
        else:
            mono_np = mono.numpy()

        return (np.clip(mono_np, -1.0, 1.0) * 32767.0).astype(np.int16)

    def separate_int16(self, audio_int16: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
        wav_f = audio_int16.astype(np.float32) / 32768.0
        return self.separate_waveform(wav_f, sr=sr, target_sr=sr)


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

        outputs_meta = self.session.get_outputs()
        self.is_frame_level = bool(
            outputs_meta and outputs_meta[0].shape and len(outputs_meta[0].shape) == 3
        )

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

    def predict_frame_probs(
        self, wav_float32: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return `(root_prob, qual_prob, ext_prob, bass_prob)` of shape `(T, C)` for frame-level models."""
        inp = wav_float32[np.newaxis, :]
        outputs = self.session.run(None, {self.input_name: inp})
        r_logits, q_logits, e_logits = outputs[0], outputs[1], outputs[2]
        root_prob = _softmax(r_logits[0], axis=-1)
        qual_prob = _softmax(q_logits[0], axis=-1)
        ext_prob = _softmax(e_logits[0], axis=-1)
        if len(outputs) >= 4:
            bass_prob = _softmax(outputs[3][0], axis=-1)
        else:
            bass_prob = np.zeros((len(root_prob), 13), dtype=np.float32)
            bass_prob[:, 12] = 1.0
        return root_prob, qual_prob, ext_prob, bass_prob

    def decode_frame_probs(
        self,
        root_prob: np.ndarray,
        qual_prob: np.ndarray,
        ext_prob: np.ndarray,
        bass_prob: np.ndarray,
    ) -> np.ndarray:
        root_ids = root_prob.argmax(axis=-1)
        root_confs = root_prob[np.arange(len(root_ids)), root_ids]

        qual_ids = qual_prob.argmax(axis=-1)
        qual_confs = qual_prob[np.arange(len(qual_ids)), qual_ids]

        if self.combo_mask is not None:
            ext_masked = ext_prob.copy()
            mask = self.combo_mask[qual_ids]
            ext_masked[~mask] = 0.0
            s = ext_masked.sum(axis=-1, keepdims=True)
            ext_masked = np.where(s > 0, ext_masked / np.maximum(s, 1e-12), ext_masked)
            ext_ids = ext_masked.argmax(axis=-1)
            ext_confs = ext_masked[np.arange(len(ext_ids)), ext_ids]
        else:
            ext_ids = ext_prob.argmax(axis=-1)
            ext_confs = ext_prob[np.arange(len(ext_ids)), ext_ids]

        bass_ids = bass_prob.argmax(axis=-1)
        avg_confs = weighted_chord_conf(root_confs, qual_confs, ext_confs)
        return np.column_stack([root_ids, qual_ids, ext_ids, bass_ids, avg_confs])

    def predict(self, wav_float32: np.ndarray):
        inp = wav_float32[np.newaxis, :]
        outputs = self.session.run(None, {self.input_name: inp})
        r_logits, q_logits, e_logits = outputs[0], outputs[1], outputs[2]

        if r_logits.ndim == 3:
            root_prob = _softmax(r_logits[0], axis=-1)
            qual_prob = _softmax(q_logits[0], axis=-1)
            ext_prob = _softmax(e_logits[0], axis=-1)
            if len(outputs) >= 4:
                bass_prob = _softmax(outputs[3][0], axis=-1)
            else:
                bass_prob = np.zeros((len(root_prob), 13), dtype=np.float32)
                bass_prob[:, 12] = 1.0
            return self.decode_frame_probs(root_prob, qual_prob, ext_prob, bass_prob)

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
        else:
            bass_id = 12

        avg_conf = float(weighted_chord_conf(root_conf, qual_conf, ext_conf))
        return root_id, qual_id, ext_id, bass_id, avg_conf


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


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


def _decode_raw_waveform(data: bytes, filename: str = "") -> tuple[np.ndarray, int]:
    """Decode raw audio bytes preserving stereo channels `(C, N)` float32 in `[-1, 1]` + original SR."""
    ext = Path(filename).suffix.lower() if filename else ""
    if ext in ("", ".wav", ".flac", ".ogg"):
        try:
            import soundfile as sf
            audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
            return audio.T, int(sr)
        except Exception as exc:
            log.debug(f"soundfile decode failed ({exc}), falling back to librosa")

    import librosa
    wav, sr_orig = librosa.load(io.BytesIO(data), sr=None, mono=False)
    if wav.ndim == 1:
        wav = np.stack([wav, wav], axis=0)
    return wav.astype(np.float32), int(sr_orig)


def load_audio_bytes(
    data: bytes,
    target_sr: int,
    filename: str = "",
    separate: bool = False,
    separator: HarmonicSeparator | None = None,
) -> np.ndarray:
    if separate and separator is not None:
        wav_stereo, sr_orig = _decode_raw_waveform(data, filename=filename)
        return separator.separate_waveform(wav_stereo, sr=sr_orig, target_sr=target_sr)

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
            if np.abs(wav).max() > 1.0:
                wav = wav / 32768.0
            if sr != target_sr:
                ta = importlib.import_module("torchaudio")
                torch = importlib.import_module("torch")
                wav_t = torch.from_numpy(wav).unsqueeze(0)
                wav_t = ta.functional.resample(wav_t, sr, target_sr)
                wav = wav_t.squeeze(0).numpy()
            return (wav * 32767).clip(-32768, 32767).astype(np.int16)
        except Exception as exc:
            log.debug(f"scipy WAV decode failed: {exc}, falling back to librosa")

    try:
        import librosa
        wav, sr_orig = librosa.load(io.BytesIO(data), sr=None, mono=True)
        if sr_orig != target_sr:
            wav = librosa.resample(wav, orig_sr=sr_orig, target_sr=target_sr)
        return (wav * 32767).clip(-32768, 32767).astype(np.int16)
    except ImportError:
        raise RuntimeError("librosa not installed: pip install librosa")


def _triangle_weights(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones(max(1, n), dtype=np.float32)
    mid = (n - 1) / 2.0
    w = 1.0 - np.abs(np.arange(n, dtype=np.float32) - mid) / (mid + 1.0)
    return np.maximum(w, 0.1)


def _infer_frame_level(
    predictor: ONNXPredictor,
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    crop_s: int = CROP_S,
    step_s: float = STEP_S,
    rms_threshold_db: float = RMS_THRESHOLD_DB,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    min_duration_ms: float = MIN_DURATION_MS,
) -> list[dict[str, Any]]:
    crop_samples = crop_s * sample_rate
    step_samples = max(MERT_HOP_SAMPLES, int(round(step_s * sample_rate / MERT_HOP_SAMPLES)) * MERT_HOP_SAMPLES)
    audio_len = len(audio)
    duration_ms = round(audio_len / sample_rate * 1000.0, 1)

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

    use_prob_fusion = isinstance(predictor, ONNXPredictor)
    expected_crop_frames = mert_output_length(crop_samples)
    total_frames = max(1, mert_output_length(max(audio_len, crop_samples)))
    if audio_len < crop_samples:
        total_frames = max(1, int(np.ceil(audio_len / MERT_HOP_SAMPLES)))

    if use_prob_fusion:
        acc_r = np.zeros((total_frames, 12), dtype=np.float32)
        acc_q = np.zeros((total_frames, 6), dtype=np.float32)
        acc_e = np.zeros((total_frames, 15), dtype=np.float32)
        acc_b = np.zeros((total_frames, 13), dtype=np.float32)
        acc_w = np.zeros(total_frames, dtype=np.float32)

        for pos, crop in windows:
            if not _rms_gate(crop, rms_threshold_db):
                continue
            wav = _normalize(crop)
            rp, qp, ep, bp = predictor.predict_frame_probs(wav)
            n_f = len(rp)
            tw = _triangle_weights(n_f)
            start_f = int(round(pos / float(MERT_HOP_SAMPLES)))
            end_f = min(total_frames, start_f + n_f)
            span = end_f - start_f
            if span <= 0:
                continue
            w_col = tw[:span, None]
            acc_r[start_f:end_f] += rp[:span] * w_col
            acc_q[start_f:end_f] += qp[:span] * w_col
            acc_e[start_f:end_f] += ep[:span] * w_col
            acc_b[start_f:end_f] += bp[:span] * w_col
            acc_w[start_f:end_f] += tw[:span]

        active_mask = acc_w > 0
        decoded = np.zeros((total_frames, 5), dtype=np.float32)
        decoded[:, 0:3] = -1
        decoded[:, 3] = 12
        if active_mask.any():
            inv_w = 1.0 / acc_w[active_mask, None]
            decoded[active_mask] = predictor.decode_frame_probs(
                acc_r[active_mask] * inv_w,
                acc_q[active_mask] * inv_w,
                acc_e[active_mask] * inv_w,
                acc_b[active_mask] * inv_w,
            )

        frame_ms = MERT_HOP_SAMPLES / float(sample_rate) * 1000.0
        frame_results: list[dict[str, Any]] = []
        for g in range(total_frames):
            silent = not bool(active_mask[g])
            if silent:
                r, q, e, b, c = -1, -1, -1, 12, 0.0
            else:
                r, q, e, b, c = (
                    int(decoded[g, 0]),
                    int(decoded[g, 1]),
                    int(decoded[g, 2]),
                    int(decoded[g, 3]),
                    float(decoded[g, 4]),
                )
                if c < confidence_threshold:
                    r, q, e, b = -1, -1, -1, 12
            s_ms = round(g * frame_ms, 1)
            e_ms = duration_ms if g == total_frames - 1 else round((g + 1) * frame_ms, 1)
            frame_results.append({
                "start_ms": s_ms,
                "end_ms": e_ms,
                "root_id": r,
                "quality_id": q,
                "ext_id": e,
                "bass_id": b,
                "chord": chord_name(r, q, e, b),
                "confidence": round(c, 4),
                "silent": silent,
            })

        min_frames = max(1, int(round(min_duration_ms / frame_ms)))
        smoothed = _smooth_jitter(frame_results, max_jitter_frames=min_frames)
        merged = _merge_consecutive(smoothed)
        if merged:
            merged[0]["start_ms"] = 0.0
            merged[-1]["end_ms"] = duration_ms
        return merged

    # Fallback path for custom/mocked predictors returning `(T, 5)` directly
    frame_data: dict[int, dict[str, Any]] = {}
    observed_n_frames = expected_crop_frames

    for pos, crop in windows:
        silent = not _rms_gate(crop, rms_threshold_db)
        if silent:
            n_frames = observed_n_frames
            start_frame = int(round(pos / float(crop_samples) * n_frames))
            for t in range(n_frames):
                g = start_frame + t
                if g not in frame_data:
                    frame_data[g] = {
                        "root_id": -1, "quality_id": -1, "ext_id": -1,
                        "bass_id": 12, "conf": 0.0, "silent": True,
                    }
        else:
            wav = _normalize(crop)
            preds = predictor.predict(wav)
            n_frames = len(preds)
            observed_n_frames = n_frames
            start_frame = int(round(pos / float(crop_samples) * n_frames))
            for t in range(n_frames):
                g = start_frame + t
                r, q, e, b, c = preds[t]
                r, q, e, b = int(r), int(q), int(e), int(b)
                c = float(c)
                if c < confidence_threshold:
                    r, q, e, b = -1, -1, -1, 12
                if g not in frame_data or frame_data[g]["silent"] or c > frame_data[g]["conf"]:
                    frame_data[g] = {
                        "root_id": r, "quality_id": q, "ext_id": e,
                        "bass_id": b, "conf": c, "silent": False,
                    }

    max_g = max(frame_data.keys()) + 1 if frame_data else expected_crop_frames
    step_ms = duration_ms / float(max(1, max_g))
    frame_results = []
    for g in range(max_g):
        fd = frame_data.get(g)
        if fd is None:
            r, q, e, b, c, s = -1, -1, -1, 12, 0.0, True
        else:
            r, q, e, b, c, s = fd["root_id"], fd["quality_id"], fd["ext_id"], fd["bass_id"], fd["conf"], fd["silent"]
        frame_results.append({
            "start_ms": round(g * step_ms, 1),
            "end_ms": duration_ms if g == max_g - 1 else round((g + 1) * step_ms, 1),
            "root_id": r,
            "quality_id": q,
            "ext_id": e,
            "bass_id": b,
            "chord": chord_name(r, q, e, b),
            "confidence": round(c, 4),
            "silent": s,
        })

    min_frames = max(1, int(round(min_duration_ms / max(step_ms, 1.0))))
    smoothed = _smooth_jitter(frame_results, max_jitter_frames=min_frames)
    merged = _merge_consecutive(smoothed)
    if merged:
        merged[0]["start_ms"] = 0.0
        merged[-1]["end_ms"] = duration_ms
    return merged


def infer_audio(
    predictor: ONNXPredictor,
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    crop_s: int = CROP_S,
    step_s: float = STEP_S,
    rms_threshold_db: float = RMS_THRESHOLD_DB,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> list[dict[str, Any]]:
    if getattr(predictor, "is_frame_level", False):
        return _infer_frame_level(
            predictor,
            audio,
            sample_rate=sample_rate,
            crop_s=crop_s,
            step_s=step_s,
            rms_threshold_db=rms_threshold_db,
            confidence_threshold=confidence_threshold,
        )

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
_separator: HarmonicSeparator | None = None


def get_predictor() -> ONNXPredictor:
    global _predictor
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _predictor


def get_separator() -> HarmonicSeparator:
    global _separator
    if _separator is None:
        _separator = HarmonicSeparator(model_name=DEMUCS_MODEL, segment=DEMUCS_SEGMENT)
    return _separator


def create_app() -> "FastAPI":
    if not _FASTAPI_AVAILABLE:
        raise RuntimeError("fastapi not installed: pip install fastapi uvicorn")

    app = FastAPI(title="ChordSense Pro — Chord Recognition API", version="1.1.0")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": MODEL_PATH,
            "demucs_model": DEMUCS_MODEL,
            "demucs_default": USE_DEMUCS_DEFAULT,
        }

    @app.post("/predict")
    async def predict(
        file: UploadFile = File(...),
        step_s: float = Query(default=STEP_S, ge=0.1, le=5.0),
        rms_threshold_db: float = Query(default=RMS_THRESHOLD_DB),
        confidence_threshold: float = Query(default=CONFIDENCE_THRESHOLD, ge=0.0, le=1.0),
        separate: bool = Query(default=USE_DEMUCS_DEFAULT, description="Strip vocals & drums via htdemucs_ft"),
    ):
        if not (file.filename or "").lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
            raise HTTPException(status_code=400, detail="Unsupported file format")

        data = await file.read()
        t_start = time.perf_counter()
        try:
            sep = get_separator() if separate else None
            audio = load_audio_bytes(
                data, SAMPLE_RATE, file.filename or "", separate=separate, separator=sep
            )
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Audio decode/separation failed: {exc}")

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
            "separated": separate,
            "demucs_model": DEMUCS_MODEL if separate else None,
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
    parser.add_argument("--no-demucs-preload", action="store_true", help="Lazy-load htdemucs_ft on first request")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    global _predictor
    _model_path = args.model

    if not Path(_model_path).exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    log.info(f"Loading ONNX model: {_model_path}")
    _predictor = ONNXPredictor(_model_path, args.classes_json)
    if USE_DEMUCS_DEFAULT and not args.no_demucs_preload:
        get_separator()
    log.info("Model loaded. Starting server...")

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)


if __name__ == "__main__":
    main()
