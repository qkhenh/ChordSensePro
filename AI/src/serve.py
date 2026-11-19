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

from src.crf import viterbi_crf_numpy
from src.music_theory import (
    FLAT_KEY_IDS,
    N_SKELETONS,
    N_TYPES,
    REL_PITCH_INDEX,
    ROOT_SILENCE_ID,
    SKEL_BASE_TYPE,
    SKEL_TYPE_MASK,
    TYPE_TENSION_REL_MASK,
    TYPE_TO_QE,
)

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
NORM_STD_FLOOR: float = 1.8e-3  # -55 dBFS floor preventing silent hiss amplification


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
RMS_THRESHOLD_DB = float(os.environ.get("SERVE_RMS_THRESHOLD_DB", "-45.0"))
CONFIDENCE_THRESHOLD = float(os.environ.get("SERVE_CONFIDENCE_THRESHOLD", "0.30"))
MIN_DURATION_MS = float(os.environ.get("SERVE_MIN_DURATION_MS", "300.0"))
MIN_EXT_DURATION_MS = float(os.environ.get("SERVE_MIN_EXT_DURATION_MS", "500.0"))
SLASH_BASS_THRESHOLD = float(os.environ.get("SERVE_SLASH_BASS_THRESHOLD", "0.50"))
RARE_EXT_THRESHOLD = float(os.environ.get("SERVE_RARE_EXT_THRESHOLD", "0.48"))
ROOT_SWITCH_PENALTY = float(os.environ.get("SERVE_ROOT_SWITCH_PENALTY", "5.5"))
QUAL_SWITCH_PENALTY = float(os.environ.get("SERVE_QUAL_SWITCH_PENALTY", "3.5"))
BASS_SWITCH_PENALTY = float(os.environ.get("SERVE_BASS_SWITCH_PENALTY", "3.5"))
EXT_SWITCH_PENALTY = float(os.environ.get("SERVE_EXT_SWITCH_PENALTY", "6.0"))
AUTO_KEY_SPELLING = os.environ.get("SERVE_AUTO_KEY_SPELLING", "1").lower() in ("1", "true", "yes")
USE_DEMUCS_DEFAULT = os.environ.get("SERVE_USE_DEMUCS", "1").lower() in ("1", "true", "yes")
DEMUCS_MODE_DEFAULT = os.environ.get("SERVE_DEMUCS_MODE", "auto").lower()  # "auto" | "always" | "off"
DEMUCS_MODEL = os.environ.get("SERVE_DEMUCS_MODEL", "htdemucs_ft")
DEMUCS_SEGMENT = float(os.environ.get("SERVE_DEMUCS_SEGMENT", "6.0"))
HOST = os.environ.get("SERVE_HOST", "0.0.0.0")
PORT = int(os.environ.get("SERVE_PORT", "8000"))
MODEL_PATH = os.environ.get("SERVE_MODEL_PATH", "results/model_v5_p3.onnx")
CLASSES_JSON = os.environ.get("SERVE_CLASSES_JSON", "../DE/data/export/v3/classes.json")

ROOT_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ROOT_NAMES_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
KEY_DISPLAY_NAMES = [f"{n} major" for n in ROOT_NAMES_FLAT] + [f"{n}m" for n in ROOT_NAMES_FLAT]
ULTRA_RARE_EXT_IDS: tuple[int, ...] = (1, 7, 9, 10, 11, 12, 13, 14)

# Krumhansl-Kessler key profiles for automatic sharp/flat enharmonic spelling (v4 fallback)
_KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float64)
_KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float64)
_FLAT_MAJOR_ROOTS = {5, 10, 3, 8, 1}   # F, Bb, Eb, Ab, Db
_FLAT_MINOR_ROOTS = {2, 7, 0, 5, 10}   # Dm, Gm, Cm, Fm, Bbm
_QUALITY_INTERVALS: dict[int, tuple[int, ...]] = {
    0: (0, 4, 7),  # maj
    1: (0, 3, 7),  # min
    2: (0, 4, 7),  # dom
    3: (0, 3, 6),  # dim
    4: (0, 2, 7),  # sus2
    5: (0, 5, 7),  # sus4
}
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


def canonical_bass_id(root_id: int, bass_id: int) -> int:
    """Normalize root-position bass (`bass == root` or out-of-range) to `12` (`none`)."""
    if root_id < 0 or root_id >= 12 or bass_id < 0 or bass_id >= 12 or bass_id == root_id:
        return 12
    return int(bass_id)


def canonicalize_chord_tuple(
    root_id: int, quality_id: int, ext_id: int, bass_id: int = 12
) -> tuple[int, int, int, int]:
    """Canonicalize root-bass identity and symmetric diminished-7th slash inversions."""
    if root_id < 0 or root_id >= 12:
        return -1, -1, -1, 12
    b = canonical_bass_id(root_id, bass_id)
    # Symmetric dim7: Cdim7 == Ebdim7/C == F#dim7/C == Adim7/C -> spell with bass as root
    if quality_id == 3 and ext_id == 5 and 0 <= b < 12 and (root_id - b) % 3 == 0:
        return b, 3, 5, 12
    return int(root_id), int(quality_id), int(ext_id), b


def chord_name(
    root_id: int,
    quality_id: int,
    ext_id: int,
    bass_id: int = 12,
    prefer_flats: bool = False,
) -> str:
    if root_id < 0 or root_id >= 12:
        return "N"
    names = ROOT_NAMES_FLAT if prefer_flats else ROOT_NAMES
    r = names[root_id]
    suffix = CHORD_SUFFIX.get((quality_id, ext_id))
    if suffix is None:
        q = QUALITY_NAMES[quality_id]
        e = EXT_NAMES[ext_id]
        suffix = QUALITY_SUFFIX[q] + EXT_SUFFIX[e]
    name = r + suffix
    if 0 <= bass_id < 12 and bass_id != root_id:
        name += "/" + names[bass_id]
    return name


def weighted_chord_conf(
    root_conf: float | np.ndarray,
    qual_conf: float | np.ndarray,
    ext_conf: float | np.ndarray,
) -> float | np.ndarray:
    """Compute chord confidence from root/quality/ext only (excluding dominant `bass=none`)."""
    return 0.40 * root_conf + 0.35 * qual_conf + 0.25 * ext_conf


def detect_solo_clean_audio(wav_stereo_or_mono: np.ndarray, sr: int) -> bool:
    """Fast (~30ms) detector for solo piano / clean harmonic instrument (no drums & no vocal fricatives).

    Uses magnitude-domain HPSS median filtering (`perc_mag < 0.34`) and `4.5-10 kHz` sibilance/cymbal
    magnitude ratio (`hf_mag < 0.008`), calibrated so solo piano bypasses Demucs while full-band mixes
     ALWAYS run `htdemucs_ft`.
    """
    try:
        from scipy.ndimage import median_filter
        from scipy.signal import stft

        mono = wav_stereo_or_mono.mean(axis=0) if wav_stereo_or_mono.ndim > 1 else wav_stereo_or_mono
        mono = mono.astype(np.float32)
        if len(mono) < sr * 2:
            return False

        win_len = min(len(mono), sr * 8)
        if len(mono) > win_len:
            hop_scan = max(1, sr)
            best_pos, best_e = 0, -1.0
            for p in range(0, len(mono) - win_len + 1, hop_scan):
                e = float(np.mean(mono[p: p + win_len] ** 2))
                if e > best_e:
                    best_e = e
                    best_pos = p
            clip = mono[best_pos: best_pos + win_len]
        else:
            clip = mono

        if float(np.max(np.abs(clip))) < 1e-3:
            return True

        freqs, _, zxx = stft(clip, fs=sr, nperseg=2048, noverlap=1536)
        mag = np.abs(zxx)
        harm = median_filter(mag, size=(1, 17))
        perc = median_filter(mag, size=(17, 1))
        perc_mag = float(perc.sum()) / float(harm.sum() + perc.sum() + 1e-8)

        hf_mask = (freqs >= 4500.0) & (freqs <= min(10000.0, sr * 0.48))
        hf_mag = float(mag[hf_mask].sum()) / float(mag.sum() + 1e-8) if hf_mask.any() else 0.0

        return bool(perc_mag < 0.34 and hf_mag < 0.008)
    except Exception as exc:
        log.debug(f"detect_solo_clean_audio fallback ({exc})")
        return False


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
                    progress=True,
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

        # Detect v5 2-stage ONNX (`encoder.onnx` + companion `encoder_decoder.onnx`)
        p = Path(model_path)
        meta_path = p.with_suffix(".meta.json")
        self.v5_meta: dict[str, Any] = {}
        if meta_path.exists():
            try:
                self.v5_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                self.v5_meta = {}

        dec_name = self.v5_meta.get("decoder_onnx", f"{p.stem}_decoder{p.suffix}")
        dec_path = p.with_name(dec_name)
        self.decoder_session = None
        self.is_v5 = False
        if dec_path.exists():
            self.decoder_session = ort.InferenceSession(str(dec_path), sess_options=opts, providers=providers)
            self.is_v5 = True
            self.is_frame_level = True
            self.crf_stride = int(self.v5_meta.get("crf_stride", 4))
            self.context_dim = int(self.v5_meta.get("context_dim", 256))
            log.info(f"ONNX v5 Full-Song Context Decoder loaded: {dec_path}")
        else:
            self.is_frame_level = bool(
                outputs_meta and outputs_meta[0].shape and len(outputs_meta[0].shape) == 3
            )
            self.crf_stride = 4
            self.context_dim = 256

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

    def encode_chunk_v5(self, wav_float32: np.ndarray) -> np.ndarray:
        """Run v5 acoustic encoder on a normalized 5s chunk `(N,)` -> `(T_chunk, 256)`."""
        inp = wav_float32[np.newaxis, :].astype(np.float32)
        emb = self.session.run(None, {self.input_name: inp})[0]
        return emb[0]

    def decode_song_v5(self, song_emb: np.ndarray, frame_rms_db: np.ndarray) -> dict[str, np.ndarray]:
        """Run v5 context decoder on `(T_song, 256)` (chunking smoothly if song > 75s)."""
        if self.decoder_session is None:
            raise RuntimeError("v5 decoder_session is not initialized")

        total_frames, c_dim = song_emb.shape
        max_win_frames = 4500   # ~60s @ 75 Hz
        step_frames = 2252      # ~30s step (divisible by crf_stride=4)
        stride = self.crf_stride

        if total_frames <= int(max_win_frames * 1.25):
            outs = self.decoder_session.run(
                None,
                {
                    "emb": song_emb[np.newaxis, :, :].astype(np.float32),
                    "frame_rms_db": frame_rms_db[np.newaxis, :].astype(np.float32),
                },
            )
            return {
                "crf_emissions": outs[0][0],
                "crf_transitions": outs[1],
                "crf_start": outs[2],
                "crf_end": outs[3],
                "type_logits_by_root": outs[4][0],
                "bass_logits": outs[5][0],
                "key_logits": outs[6][0],
                "sustain_env": outs[7][0],
            }

        # Overlap-add for very long tracks (>75s, e.g. 5-min Jazz recordings)
        total_crf = int(np.ceil(total_frames / float(stride)))
        acc_crf = np.zeros((total_crf, 1 + 12 * N_SKELETONS), dtype=np.float32)
        acc_w_crf = np.zeros(total_crf, dtype=np.float32)
        acc_type = np.zeros((total_frames, 12, N_TYPES), dtype=np.float32)
        acc_bass = np.zeros((total_frames, 13), dtype=np.float32)
        acc_key = np.zeros((total_frames, 24), dtype=np.float32)
        acc_sust = np.zeros((total_frames, 12), dtype=np.float32)
        acc_w = np.zeros(total_frames, dtype=np.float32)

        crf_trans = crf_start = crf_end = None
        pos = 0
        starts = []
        while pos + max_win_frames <= total_frames:
            starts.append(pos)
            pos += step_frames
        if not starts or starts[-1] + max_win_frames < total_frames:
            last_s = max(0, ((total_frames - max_win_frames) // stride) * stride)
            if not starts or last_s != starts[-1]:
                starts.append(last_s)

        for s in starts:
            e = min(total_frames, s + max_win_frames)
            outs = self.decoder_session.run(
                None,
                {
                    "emb": song_emb[np.newaxis, s:e, :].astype(np.float32),
                    "frame_rms_db": frame_rms_db[np.newaxis, s:e].astype(np.float32),
                },
            )
            crf_em, crf_trans, crf_start, crf_end = outs[0][0], outs[1], outs[2], outs[3]
            tw = _triangle_weights(e - s)
            acc_type[s:e] += outs[4][0] * tw[:, None, None]
            acc_bass[s:e] += outs[5][0] * tw[:, None]
            acc_key[s:e] += outs[6][0] * tw[:, None]
            acc_sust[s:e] += outs[7][0] * tw[:, None]
            acc_w[s:e] += tw

            cs = s // stride
            ce = min(total_crf, cs + len(crf_em))
            tw_c = _triangle_weights(ce - cs)
            acc_crf[cs:ce] += crf_em[: ce - cs] * tw_c[:, None]
            acc_w_crf[cs:ce] += tw_c

        inv_w = 1.0 / np.maximum(acc_w, 1e-6)
        inv_wc = 1.0 / np.maximum(acc_w_crf, 1e-6)
        return {
            "crf_emissions": acc_crf * inv_wc[:, None],
            "crf_transitions": crf_trans,
            "crf_start": crf_start,
            "crf_end": crf_end,
            "type_logits_by_root": acc_type * inv_w[:, None, None],
            "bass_logits": acc_bass * inv_w[:, None],
            "key_logits": acc_key * inv_w[:, None],
            "sustain_env": acc_sust * inv_w[:, None],
        }

    def predict_frame_probs(
        self, wav_float32: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return `(root_prob, qual_prob, ext_prob, bass_prob)` of shape `(T, C)` for v4 frame-level models."""
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
        slash_bass_threshold: float = SLASH_BASS_THRESHOLD,
    ) -> np.ndarray:
        n = len(root_prob)
        idx = np.arange(n)
        root_ids = root_prob.argmax(axis=-1)
        root_confs = root_prob[idx, root_ids]

        qual_ids = qual_prob.argmax(axis=-1)
        qual_confs = qual_prob[idx, qual_ids]

        if self.combo_mask is not None:
            ext_masked = ext_prob.copy()
            mask = self.combo_mask[qual_ids]
            ext_masked[~mask] = 0.0
            s = ext_masked.sum(axis=-1, keepdims=True)
            ext_masked = np.where(s > 0, ext_masked / np.maximum(s, 1e-12), ext_masked)
            ext_ids = ext_masked.argmax(axis=-1)
            ext_confs = ext_masked[idx, ext_ids]
        else:
            ext_ids = ext_prob.copy().argmax(axis=-1)
            ext_confs = ext_prob[idx, ext_ids]

        bp = bass_prob.copy()
        if bp.shape[-1] >= 13 and n > 0:
            bp[idx, 12] += bp[idx, root_ids]
            bp[idx, root_ids] = 0.0
            slash_best = bp[:, :12].argmax(axis=-1)
            slash_conf = bp[idx, slash_best]
            use_slash = (slash_conf >= slash_bass_threshold) & (slash_conf > bp[:, 12])
            bass_ids = np.where(use_slash, slash_best, 12)
        else:
            bass_ids = np.full(n, 12, dtype=np.int64)

        for t in range(n):
            r_c, q_c, e_c, b_c = canonicalize_chord_tuple(
                int(root_ids[t]), int(qual_ids[t]), int(ext_ids[t]), int(bass_ids[t])
            )
            root_ids[t], qual_ids[t], ext_ids[t], bass_ids[t] = r_c, q_c, e_c, b_c

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
    """Soft-clamped normalization (`-55 dBFS` floor) preventing quiet room hiss from exploding."""
    wav = wav.astype(np.float32) / 32768.0
    denom = max(float(wav.std()), NORM_STD_FLOOR)
    return ((wav - float(wav.mean())) / denom).astype(np.float32)


def _compute_song_frame_rms_db(audio_int16: np.ndarray, n_frames: int) -> np.ndarray:
    """Compute pre-normalization dBFS per MERT frame `(n_frames,)` across the full song."""
    if n_frames <= 0:
        return np.empty(0, dtype=np.float32)
    needed = n_frames * MERT_HOP_SAMPLES
    wav = audio_int16[:needed].astype(np.float32) / 32768.0
    if len(wav) < needed:
        wav = np.pad(wav, (0, needed - len(wav)))
    frames = wav.reshape(n_frames, MERT_HOP_SAMPLES)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    rms_s = np.convolve(np.pad(rms, (1, 1), mode="edge"), np.array([0.25, 0.5, 0.25], dtype=np.float32), mode="valid")
    return 20.0 * np.log10(np.maximum(rms_s, 1e-6))


def _compute_decay_lock_scale(rms_db_crf: np.ndarray) -> np.ndarray:
    """Compute Onset-Decay Lock multiplier (`>= 1.0`) per CRF step to freeze decaying chord tails."""
    n = len(rms_db_crf)
    if n <= 2:
        return np.ones(n, dtype=np.float32)
    scale = np.ones(n, dtype=np.float32)
    peak_db = float(rms_db_crf[0])
    for t in range(1, n):
        cur = float(rms_db_crf[t])
        diff = cur - float(rms_db_crf[t - 1])
        if diff > 1.8:
            # New chord attack / note strike -> reset peak, unlock transitions
            peak_db = cur
            scale[t] = 1.0
        else:
            peak_db = max(peak_db, cur)
            drop_db = peak_db - cur
            if diff <= 0.3 and drop_db >= 6.0:
                scale[t] = min(3.0, 1.0 + (drop_db - 6.0) / 8.0)
    return scale


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
    demucs_mode: str = DEMUCS_MODE_DEFAULT,
) -> np.ndarray:
    """Decode audio to `target_sr` mono `int16`, with automatic solo-instrument Demucs bypass in `'auto'` mode."""
    mode = (demucs_mode or "auto").lower()
    if separate and mode != "off" and separator is not None:
        wav_stereo, sr_orig = _decode_raw_waveform(data, filename=filename)
        if mode == "auto" and detect_solo_clean_audio(wav_stereo, sr_orig):
            log.info("Auto-detected clean solo instrument (no drums/vocals) — bypassing Demucs")
            mono = wav_stereo.mean(axis=0).astype(np.float32)
            if sr_orig != target_sr:
                if _TORCHAUDIO_AVAILABLE:
                    import torch
                    mono = torchaudio.functional.resample(
                        torch.from_numpy(mono).unsqueeze(0), sr_orig, target_sr
                    ).squeeze(0).numpy()
                else:
                    import librosa
                    mono = librosa.resample(mono, orig_sr=sr_orig, target_sr=target_sr)
            return (np.clip(mono, -1.0, 1.0) * 32767.0).astype(np.int16)

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


def _smooth_prob_sequence(prob: np.ndarray, kernel_frames: int = 7) -> np.ndarray:
    """Smooth `(T, C)` probability trajectory with a symmetric triangular window."""
    n = len(prob)
    if n <= 2 or kernel_frames <= 1:
        return prob.copy()
    k = min(kernel_frames, n if n % 2 == 1 else max(1, n - 1))
    if k <= 1:
        return prob.copy()
    half = k // 2
    kernel = (half + 1 - np.abs(np.arange(k, dtype=np.float32) - half))
    kernel /= kernel.sum()
    padded = np.pad(prob, ((half, half), (0, 0)), mode="edge")
    out = np.empty_like(prob, dtype=np.float32)
    for c in range(prob.shape[1]):
        out[:, c] = np.convolve(padded[:, c], kernel, mode="valid")
    s = out.sum(axis=-1, keepdims=True)
    return np.where(s > 0, out / np.maximum(s, 1e-12), prob)


def _compute_onset_weights(audio: np.ndarray, n_frames: int) -> np.ndarray:
    """Give ~2x voting weight to chord attacks (strums/downbeats) vs decaying arpeggio tails."""
    if n_frames <= 0 or len(audio) == 0:
        return np.ones(max(1, n_frames), dtype=np.float32)
    hop = MERT_HOP_SAMPLES
    needed = n_frames * hop
    wav = audio[:needed].astype(np.float32) / 32768.0
    if len(wav) < needed:
        wav = np.pad(wav, (0, needed - len(wav)))
    frames = wav.reshape(n_frames, hop)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-10)
    flux = np.maximum(0.0, np.diff(rms, prepend=rms[0]))
    p90 = float(np.percentile(flux, 90)) if len(flux) > 4 else float(flux.max())
    norm_flux = np.clip(flux / max(p90, 1e-5), 0.0, 1.5)
    attack_env = np.zeros(n_frames, dtype=np.float32)
    cur = 0.0
    decay = 0.88
    for i in range(n_frames):
        cur = max(float(norm_flux[i]), cur * decay)
        attack_env[i] = cur
    return 1.0 + attack_env


def _viterbi_root_quality(
    root_prob: np.ndarray,
    qual_prob: np.ndarray,
    root_switch_penalty: float = ROOT_SWITCH_PENALTY,
    qual_switch_penalty: float = QUAL_SWITCH_PENALTY,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact 72-state `(root, quality)` Sticky Viterbi in O(T * 72) vectorized NumPy."""
    n = len(root_prob)
    if n == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    log_r = np.log(np.clip(root_prob, 1e-7, 1.0))
    log_q = np.log(np.clip(qual_prob, 1e-7, 1.0))
    log_obs = (log_r[:, :, None] + 0.85 * log_q[:, None, :]).reshape(n, 72)

    pen_r = max(float(root_switch_penalty), float(qual_switch_penalty), 0.0)
    pen_q = max(float(qual_switch_penalty), 0.0)

    backptr = np.zeros((n, 72), dtype=np.int32)
    dp = log_obs[0].astype(np.float64)
    states_72 = np.arange(72, dtype=np.int32)
    root_base = np.arange(12, dtype=np.int32) * 6

    for t in range(1, n):
        best_score = dp.copy()
        best_idx = states_72.copy()

        dp_rq = dp.reshape(12, 6)
        best_q_in_r = dp_rq.argmax(axis=1)
        same_r_idx = np.repeat(root_base + best_q_in_r, 6)
        same_r_score = dp[same_r_idx] - pen_q
        use_same_r = same_r_score > best_score
        best_score = np.where(use_same_r, same_r_score, best_score)
        best_idx = np.where(use_same_r, same_r_idx, best_idx)

        g_best = int(dp.argmax())
        g_score = dp[g_best] - pen_r
        use_g = g_score > best_score
        best_score = np.where(use_g, g_score, best_score)
        best_idx = np.where(use_g, g_best, best_idx)

        backptr[t] = best_idx
        dp = best_score + log_obs[t]

    path = np.empty(n, dtype=np.int32)
    path[-1] = int(dp.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = backptr[t, path[t]]

    return (path // 6).astype(np.int64), (path % 6).astype(np.int64)


def _viterbi_sticky_1d(log_obs: np.ndarray, switch_penalty: float) -> np.ndarray:
    """Exact K-state Sticky Viterbi in O(T * K) vectorized NumPy."""
    n, k = log_obs.shape
    if n == 0:
        return np.empty(0, dtype=np.int64)
    pen = max(float(switch_penalty), 0.0)
    backptr = np.zeros((n, k), dtype=np.int32)
    dp = log_obs[0].astype(np.float64)
    states = np.arange(k, dtype=np.int32)

    for t in range(1, n):
        g_best = int(dp.argmax())
        g_score = dp[g_best] - pen
        use_switch = g_score > dp
        backptr[t] = np.where(use_switch, g_best, states)
        dp = np.where(use_switch, g_score, dp) + log_obs[t]

    path = np.empty(n, dtype=np.int64)
    path[-1] = int(dp.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = backptr[t, path[t]]
    return path


def decode_musical_sequence(
    root_prob: np.ndarray,
    qual_prob: np.ndarray,
    ext_prob: np.ndarray,
    bass_prob: np.ndarray,
    combo_mask: np.ndarray | None = None,
    onset_weights: np.ndarray | None = None,
    frame_ms: float = 13.333,
    min_ext_duration_ms: float = MIN_EXT_DURATION_MS,
    slash_bass_threshold: float = SLASH_BASS_THRESHOLD,
    rare_ext_threshold: float = RARE_EXT_THRESHOLD,
    root_switch_penalty: float = ROOT_SWITCH_PENALTY,
    qual_switch_penalty: float = QUAL_SWITCH_PENALTY,
    bass_switch_penalty: float = BASS_SWITCH_PENALTY,
    ext_switch_penalty: float = EXT_SWITCH_PENALTY,
) -> np.ndarray:
    """Decode continuous `(T, C)` probabilities into musically stable `(T, 5)` chord trajectories (v4)."""
    n = len(root_prob)
    if n == 0:
        return np.zeros((0, 5), dtype=np.float32)

    rp_s = _smooth_prob_sequence(root_prob, kernel_frames=7)
    qp_s = _smooth_prob_sequence(qual_prob, kernel_frames=7)
    ep_s = _smooth_prob_sequence(ext_prob, kernel_frames=7)
    bp_s = _smooth_prob_sequence(bass_prob, kernel_frames=7)

    ow = onset_weights if (onset_weights is not None and len(onset_weights) == n) else np.ones(n, dtype=np.float32)
    hop_scale = 13.333 / max(frame_ms, 1.0)

    root_ids, qual_ids = _viterbi_root_quality(
        rp_s,
        qp_s,
        root_switch_penalty=root_switch_penalty * hop_scale,
        qual_switch_penalty=qual_switch_penalty * hop_scale,
    )

    bass_ids = np.full(n, 12, dtype=np.int64)
    ext_ids = np.zeros(n, dtype=np.int64)
    ext_confs = np.zeros(n, dtype=np.float32)

    min_ext_frames = max(1, int(round(min_ext_duration_ms / max(frame_ms, 1.0))))

    span_start = 0
    while span_start < n:
        r_cur = int(root_ids[span_start])
        q_cur = int(qual_ids[span_start])
        span_end = span_start + 1
        while span_end < n and root_ids[span_end] == r_cur and qual_ids[span_end] == q_cur:
            span_end += 1

        bp_seg = bp_s[span_start:span_end].copy()
        if bp_seg.shape[-1] >= 13:
            bp_seg[:, 12] += bp_seg[:, r_cur]
            bp_seg[:, r_cur] = 0.0
            s_sum = bp_seg.sum(axis=-1, keepdims=True)
            bp_seg = np.where(s_sum > 0, bp_seg / np.maximum(s_sum, 1e-12), bp_seg)

            log_bp = np.log(np.clip(bp_seg, 1e-7, 1.0))
            slash_max = bp_seg[:, :12].max(axis=-1)
            weak_slash = slash_max < slash_bass_threshold
            log_bp[weak_slash, 12] += 0.75
            b_path = _viterbi_sticky_1d(log_bp, bass_switch_penalty * hop_scale)

            b_pos = 0
            seg_len = span_end - span_start
            while b_pos < seg_len:
                b_val = int(b_path[b_pos])
                b_end = b_pos + 1
                while b_end < seg_len and b_path[b_end] == b_val:
                    b_end += 1
                if b_val < 12:
                    mean_slash_p = float(bp_seg[b_pos:b_end, b_val].mean())
                    mean_rq_p = 0.5 * float(
                        rp_s[span_start + b_pos: span_start + b_end, r_cur].mean()
                        + qp_s[span_start + b_pos: span_start + b_end, q_cur].mean()
                    )
                    if mean_slash_p < slash_bass_threshold or mean_rq_p < 0.45:
                        b_path[b_pos:b_end] = 12
                b_pos = b_end
            bass_ids[span_start:span_end] = b_path

        span_start = span_end

    span_start = 0
    while span_start < n:
        r_cur = int(root_ids[span_start])
        q_cur = int(qual_ids[span_start])
        b_cur = int(bass_ids[span_start])
        span_end = span_start + 1
        while (
            span_end < n
            and root_ids[span_end] == r_cur
            and qual_ids[span_end] == q_cur
            and bass_ids[span_end] == b_cur
        ):
            span_end += 1

        ep_seg = ep_s[span_start:span_end].copy()
        ow_seg = ow[span_start:span_end]
        if combo_mask is not None:
            valid_e = combo_mask[q_cur]
            ep_seg[:, ~valid_e] = 0.0
            s_e = ep_seg.sum(axis=-1, keepdims=True)
            ep_seg = np.where(s_e > 0, ep_seg / np.maximum(s_e, 1e-12), ep_seg)

        w_sum = max(float(ow_seg.sum()), 1e-8)
        mean_ep = (ep_seg * ow_seg[:, None]).sum(axis=0) / w_sum

        gated = ep_seg.copy()
        for u_id in ULTRA_RARE_EXT_IDS:
            if u_id < gated.shape[1] and mean_ep[u_id] < rare_ext_threshold:
                gated[:, u_id] = 0.0
        row_sums = gated.sum(axis=-1, keepdims=True)
        if np.all(row_sums > 1e-8):
            ep_seg = gated / row_sums
            mean_ep = (ep_seg * ow_seg[:, None]).sum(axis=0) / w_sum

        dom_ext = int(mean_ep.argmax())
        seg_len = span_end - span_start
        if seg_len < 2 * min_ext_frames:
            e_path = np.full(seg_len, dom_ext, dtype=np.int64)
        else:
            log_ep = np.log(np.clip(ep_seg, 1e-7, 1.0))
            e_path = _viterbi_sticky_1d(log_ep, ext_switch_penalty * hop_scale)
            changed = True
            while changed:
                changed = False
                runs: list[list[int]] = []
                p = 0
                while p < seg_len:
                    q_end = p + 1
                    while q_end < seg_len and e_path[q_end] == e_path[p]:
                        q_end += 1
                    runs.append([p, q_end, int(e_path[p])])
                    p = q_end
                if len(runs) <= 1:
                    break
                shortest_i = min(range(len(runs)), key=lambda idx_r: runs[idx_r][1] - runs[idx_r][0])
                sh_s, sh_e, _ = runs[shortest_i]
                if (sh_e - sh_s) < min_ext_frames:
                    if shortest_i == 0:
                        rep_ext = runs[1][2]
                    elif shortest_i == len(runs) - 1:
                        rep_ext = runs[-2][2]
                    else:
                        left_len = runs[shortest_i - 1][1] - runs[shortest_i - 1][0]
                        right_len = runs[shortest_i + 1][1] - runs[shortest_i + 1][0]
                        rep_ext = runs[shortest_i - 1][2] if left_len >= right_len else runs[shortest_i + 1][2]
                    e_path[sh_s:sh_e] = rep_ext
                    changed = True

        ext_ids[span_start:span_end] = e_path
        ext_confs[span_start:span_end] = ep_seg[np.arange(seg_len), e_path]
        span_start = span_end

    idx_all = np.arange(n)
    root_confs = rp_s[idx_all, root_ids]
    qual_confs = qp_s[idx_all, qual_ids]
    avg_confs = weighted_chord_conf(root_confs, qual_confs, ext_confs)

    for t in range(n):
        r_c, q_c, e_c, b_c = canonicalize_chord_tuple(
            int(root_ids[t]), int(qual_ids[t]), int(ext_ids[t]), int(bass_ids[t])
        )
        root_ids[t], qual_ids[t], ext_ids[t], bass_ids[t] = r_c, q_c, e_c, b_c

    return np.column_stack([root_ids, qual_ids, ext_ids, bass_ids, avg_confs])


def _infer_v5_full_song(
    predictor: ONNXPredictor,
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    crop_s: int = CROP_S,
    step_s: float = STEP_S,
    rms_threshold_db: float = RMS_THRESHOLD_DB,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    min_duration_ms: float = MIN_DURATION_MS,
) -> list[dict[str, Any]]:
    """Run 2-Stage Model v5 Inference:
    1. Overlap-add 5s chunks in 256d embedding space (`v5_encoder.onnx`)
    2. Run Full-Song Bi-Conformer + LocalKey + Root-Conditioned Joint Head (`v5_decoder.onnx`)
    3. Decode 145-state Skeleton CRF with Silence Gate & Onset-Decay Locking
    4. Resolve Jazz Tensions (`add9, 9, #11, 13, alt`) via `sustain_env` (`>= 500ms` duration gating)
    """
    chunk_s = int(predictor.v5_meta.get("chunk_s", crop_s))
    chunk_samples = chunk_s * sample_rate
    step_samples = max(MERT_HOP_SAMPLES, int(round(step_s * sample_rate / MERT_HOP_SAMPLES)) * MERT_HOP_SAMPLES)
    audio_len = len(audio)
    duration_ms = round(audio_len / float(sample_rate) * 1000.0, 1)

    t_chunk = mert_output_length(chunk_samples)
    total_frames = max(1, mert_output_length(max(audio_len, chunk_samples)))
    if audio_len < chunk_samples:
        total_frames = max(1, int(np.ceil(audio_len / float(MERT_HOP_SAMPLES))))

    windows: list[tuple[int, np.ndarray]] = []
    if audio_len < chunk_samples:
        pad = np.zeros(chunk_samples, dtype=np.int16)
        pad[:audio_len] = audio
        windows.append((0, pad))
    else:
        pos = 0
        while pos + chunk_samples <= audio_len:
            windows.append((pos, audio[pos: pos + chunk_samples]))
            pos += step_samples
        if pos < audio_len:
            last = audio_len - chunk_samples
            if last != pos - step_samples:
                windows.append((last, audio[last: last + chunk_samples]))

    acc_emb = np.zeros((total_frames, predictor.context_dim), dtype=np.float32)
    acc_w = np.zeros(total_frames, dtype=np.float32)

    for pos, crop in windows:
        wav = _normalize(crop)
        emb = predictor.encode_chunk_v5(wav)  # (T_chunk, 256)
        n_f = len(emb)
        tw = _triangle_weights(n_f)
        start_f = int(round(pos / float(MERT_HOP_SAMPLES)))
        end_f = min(total_frames, start_f + n_f)
        span = end_f - start_f
        if span <= 0:
            continue
        acc_emb[start_f:end_f] += emb[:span] * tw[:span, None]
        acc_w[start_f:end_f] += tw[:span]

    song_emb = acc_emb / np.maximum(acc_w[:, None], 1e-6)
    raw_frame_rms_db = _compute_song_frame_rms_db(audio, total_frames)
    # Normalize `frame_rms_db` to a consistent `-1 dBFS` song peak reference (matching DE peak-normalized audio)
    # while capping max gain boost at +15 dB so pure silence files stay silent.
    peak_lin = float(np.max(np.abs(audio.astype(np.float32))) / 32768.0) if len(audio) else 0.0
    gain_offset_db = float(np.clip(-1.0 - 20.0 * np.log10(max(peak_lin, 1e-4)), 0.0, 15.0))
    frame_rms_db = raw_frame_rms_db + gain_offset_db

    dec = predictor.decode_song_v5(song_emb, frame_rms_db)
    crf_em = dec["crf_emissions"].copy()              # (T_crf, 145)
    crf_trans = dec["crf_transitions"]                # (145, 145)
    crf_start = dec["crf_start"]                      # (145,)
    crf_end = dec["crf_end"]                          # (145,)
    type_logits = dec["type_logits_by_root"]          # (T, 12, 41)
    bass_logits = dec["bass_logits"]                  # (T, 13)
    key_logits = dec["key_logits"]                    # (T, 24)
    sustain_env = dec["sustain_env"]                  # (T, 12)

    stride = predictor.crf_stride
    n_crf = len(crf_em)
    rms_crf = np.array(
        [
            float(frame_rms_db[i * stride: min(total_frames, (i + 1) * stride)].mean())
            for i in range(n_crf)
        ],
        dtype=np.float32,
    )
    active_rms = frame_rms_db[frame_rms_db > -58.0]
    p95_db = float(np.percentile(active_rms, 95)) if len(active_rms) > 8 else float(frame_rms_db.max())
    # True no-sound floor (-50 dBFS or 40 dB below peak): catches -68 dB intro hiss without chopping -46 dB piano sustain tails
    true_silence_db = min(float(rms_threshold_db), -50.0)
    dynamic_sil_floor = max(true_silence_db, p95_db - 40.0)

    quiet_steps = rms_crf < dynamic_sil_floor
    crf_em[quiet_steps, 0] += 5.0

    decay_scale = _compute_decay_lock_scale(rms_crf)
    crf_path_ds = viterbi_crf_numpy(
        crf_em,
        crf_trans,
        start_transitions=crf_start,
        end_transitions=crf_end,
        decay_lock_scale=decay_scale,
    )
    crf_path = np.repeat(crf_path_ds, stride)[:total_frames]
    if len(crf_path) < total_frames:
        crf_path = np.pad(crf_path, (0, total_frames - len(crf_path)), mode="edge")

    # Enforce hard silence gate only on true no-sound frames (< true_silence_db, e.g. -50 dBFS)
    crf_path[frame_rms_db < true_silence_db] = 0

    frame_ms = MERT_HOP_SAMPLES / float(sample_rate) * 1000.0
    min_tension_frames = max(1, int(round(MIN_EXT_DURATION_MS / max(frame_ms, 1.0))))
    hop_scale = 13.333 / max(frame_ms, 1.0)

    # Softmax probabilities over 145 CRF states -> factorized Root, Conditional Skeleton & Conditional Type confidences
    crf_frame_prob = _softmax(
        np.repeat(crf_em, stride, axis=0)[:total_frames], axis=-1
    )
    bass_prob_all = _softmax(bass_logits, axis=-1)
    key_prob_all = _softmax(key_logits, axis=-1)
    has_tension_type = TYPE_TENSION_REL_MASK.sum(axis=-1) > 0.5  # (41,)

    root_ids = np.full(total_frames, -1, dtype=np.int64)
    qual_ids = np.full(total_frames, -1, dtype=np.int64)
    ext_ids = np.full(total_frames, -1, dtype=np.int64)
    bass_ids = np.full(total_frames, 12, dtype=np.int64)
    confs = np.zeros(total_frames, dtype=np.float32)
    prefer_flats_frame = np.zeros(total_frames, dtype=bool)

    span_s = 0
    while span_s < total_frames:
        st_cur = int(crf_path[span_s])
        span_e = span_s + 1
        while span_e < total_frames and crf_path[span_e] == st_cur:
            span_e += 1

        if st_cur <= 0:
            span_s = span_e
            continue

        r_cur = (st_cur - 1) // N_SKELETONS
        sk_cur = (st_cur - 1) % N_SKELETONS
        seg_len = span_e - span_s

        # Local key spelling for this harmonic span
        mean_k = key_prob_all[span_s:span_e].mean(axis=0)
        best_key_id = int(mean_k.argmax())
        prefer_flats_frame[span_s:span_e] = best_key_id in FLAT_KEY_IDS

        # Slash bass decoding inside locked `(Root, Skeleton)` span
        bp_seg = bass_prob_all[span_s:span_e].copy()
        bp_seg[:, 12] += bp_seg[:, r_cur]
        bp_seg[:, r_cur] = 0.0
        bp_sum = bp_seg.sum(axis=-1, keepdims=True)
        bp_seg = np.where(bp_sum > 0, bp_seg / np.maximum(bp_sum, 1e-12), bp_seg)
        log_bp = np.log(np.clip(bp_seg, 1e-7, 1.0))
        log_bp[bp_seg[:, :12].max(axis=-1) < SLASH_BASS_THRESHOLD, 12] += 0.75
        b_path = _viterbi_sticky_1d(log_bp, BASS_SWITCH_PENALTY * hop_scale)
        bass_ids[span_s:span_e] = b_path

        # Within-Skeleton Jazz Tension & Sustained Color Resolution (`F` + sustained `G` -> `Fadd9`)
        allowed = SKEL_TYPE_MASK[sk_cur]  # (41,) bool
        allowed_idx = np.nonzero(allowed)[0]
        if len(allowed_idx) == 1:
            chosen_types = np.full(seg_len, int(allowed_idx[0]), dtype=np.int64)
            type_confs = np.ones(seg_len, dtype=np.float32)
        else:
            t_log = type_logits[span_s:span_e, r_cur, :].copy()  # (L, 41)
            rel_s = sustain_env[span_s:span_e][:, REL_PITCH_INDEX[r_cur]]  # (L, 12)
            tens_s = (rel_s @ TYPE_TENSION_REL_MASK.T) / np.maximum(TYPE_TENSION_REL_MASK.sum(axis=-1), 1.0)
            # Suppress transient melody notes (<500ms) from falsely activating tension types
            weak_t = (tens_s < 0.16) & has_tension_type[None, :]
            t_log = np.where(weak_t, t_log - 0.60, t_log)
            t_log[:, ~allowed] = -1e9
            t_prob = _softmax(t_log, axis=-1)

            if seg_len < 2 * min_tension_frames:
                mean_tp = t_prob.mean(axis=0)
                best_t = int(mean_tp.argmax())
                # Gate ultra-rare alterations if weak
                q_cand, e_cand = TYPE_TO_QE[best_t]
                if int(e_cand) in ULTRA_RARE_EXT_IDS and float(mean_tp[best_t]) < RARE_EXT_THRESHOLD:
                    best_t = int(SKEL_BASE_TYPE[sk_cur])
                chosen_types = np.full(seg_len, best_t, dtype=np.int64)
            else:
                log_sub = np.log(np.clip(t_prob[:, allowed_idx], 1e-7, 1.0))
                sub_path = _viterbi_sticky_1d(log_sub, EXT_SWITCH_PENALTY * 0.75 * hop_scale)
                chosen_types = allowed_idx[sub_path]
                # Absorb tension runs shorter than `min_tension_frames` (~500ms) into base/neighbor type
                changed = True
                while changed:
                    changed = False
                    runs: list[list[int]] = []
                    p = 0
                    while p < seg_len:
                        qe_i = p + 1
                        while qe_i < seg_len and chosen_types[qe_i] == chosen_types[p]:
                            qe_i += 1
                        runs.append([p, qe_i, int(chosen_types[p])])
                        p = qe_i
                    if len(runs) <= 1:
                        break
                    sh_i = min(range(len(runs)), key=lambda idx_r: runs[idx_r][1] - runs[idx_r][0])
                    rs, re, _ = runs[sh_i]
                    if (re - rs) < min_tension_frames:
                        if sh_i == 0:
                            rep_t = runs[1][2]
                        elif sh_i == len(runs) - 1:
                            rep_t = runs[-2][2]
                        else:
                            l_len = runs[sh_i - 1][1] - runs[sh_i - 1][0]
                            r_len = runs[sh_i + 1][1] - runs[sh_i + 1][0]
                            rep_t = runs[sh_i - 1][2] if l_len >= r_len else runs[sh_i + 1][2]
                        chosen_types[rs:re] = rep_t
                        changed = True

            type_confs = t_prob[np.arange(seg_len), chosen_types]

        r_lo = 1 + r_cur * N_SKELETONS
        r_hi = r_lo + N_SKELETONS
        root_confs = crf_frame_prob[span_s:span_e, r_lo:r_hi].sum(axis=-1)
        cond_skel_confs = crf_frame_prob[span_s:span_e, st_cur] / np.maximum(root_confs, 1e-6)
        root_ids[span_s:span_e] = r_cur
        qual_ids[span_s:span_e] = TYPE_TO_QE[chosen_types, 0]
        ext_ids[span_s:span_e] = TYPE_TO_QE[chosen_types, 1]
        confs[span_s:span_e] = np.clip(
            weighted_chord_conf(root_confs, cond_skel_confs, type_confs), 0.0, 1.0
        )
        span_s = span_e

    frame_results: list[dict[str, Any]] = []
    for g in range(total_frames):
        r_g = int(root_ids[g])
        silent = r_g < 0 or r_g >= 12
        if silent:
            r_c, q_c, e_c, b_c, c_val = -1, -1, -1, 12, 0.0
        else:
            r_c, q_c, e_c, b_c = canonicalize_chord_tuple(
                r_g, int(qual_ids[g]), int(ext_ids[g]), int(bass_ids[g])
            )
            c_val = float(confs[g])
            if c_val < confidence_threshold:
                r_c, q_c, e_c, b_c = -1, -1, -1, 12
                silent = True

        s_ms = round(g * frame_ms, 1)
        e_ms = duration_ms if g == total_frames - 1 else round((g + 1) * frame_ms, 1)
        use_flats = bool(prefer_flats_frame[g]) if AUTO_KEY_SPELLING else False
        frame_results.append({
            "start_ms": s_ms,
            "end_ms": e_ms,
            "root_id": r_c,
            "quality_id": q_c,
            "ext_id": e_c,
            "bass_id": b_c,
            "chord": chord_name(r_c, q_c, e_c, b_c, prefer_flats=use_flats),
            "confidence": round(c_val, 4),
            "silent": silent,
        })

    merged = _prune_short_segments(frame_results, min_duration_ms=min_duration_ms)
    if AUTO_KEY_SPELLING:
        merged = apply_key_spelling(merged)
    if merged:
        merged[0]["start_ms"] = 0.0
        merged[-1]["end_ms"] = duration_ms
    return merged


def detect_prefer_flats(segments: list[dict[str, Any]]) -> bool:
    """Estimate song key via Krumhansl-Kessler profiles to choose flat (`Bb`) vs sharp (`A#`) spelling."""
    pc_hist = np.zeros(12, dtype=np.float64)
    for seg in segments:
        r = int(seg.get("root_id", -1))
        q = int(seg.get("quality_id", -1))
        if r < 0 or r >= 12 or seg.get("silent", False):
            continue
        dur = max(0.0, float(seg["end_ms"]) - float(seg["start_ms"]))
        w = dur * max(float(seg.get("confidence", 0.5)), 0.1)
        pc_hist[r] += 1.5 * w
        for iv in _QUALITY_INTERVALS.get(q, (0, 4, 7)):
            pc_hist[(r + iv) % 12] += w

    if pc_hist.sum() <= 1e-6:
        return False

    best_score = -1e18
    best_is_flat = False
    for tonic in range(12):
        rot = np.roll(pc_hist, -tonic)
        maj_score = float(np.dot(rot, _KK_MAJOR))
        if maj_score > best_score:
            best_score = maj_score
            best_is_flat = tonic in _FLAT_MAJOR_ROOTS
        min_score = float(np.dot(rot, _KK_MINOR))
        if min_score > best_score:
            best_score = min_score
            best_is_flat = tonic in _FLAT_MINOR_ROOTS
    return best_is_flat


def apply_key_spelling(
    segments: list[dict[str, Any]], prefer_flats: bool | None = None
) -> list[dict[str, Any]]:
    if not segments:
        return segments
    use_flats = detect_prefer_flats(segments) if prefer_flats is None else bool(prefer_flats)
    out = []
    for seg in segments:
        s = seg.copy()
        r, q, e, b = canonicalize_chord_tuple(
            int(s["root_id"]),
            int(s["quality_id"]),
            int(s["ext_id"]),
            int(s.get("bass_id", 12)),
        )
        s["root_id"], s["quality_id"], s["ext_id"], s["bass_id"] = r, q, e, b
        s["chord"] = chord_name(r, q, e, b, prefer_flats=use_flats)
        out.append(s)
    return out


def _prune_short_segments(
    segments: list[dict[str, Any]],
    min_duration_ms: float = MIN_DURATION_MS,
) -> list[dict[str, Any]]:
    """Iteratively absorb micro-segments `< min_duration_ms` (including staircase `A -> B -> C`)."""
    if len(segments) <= 1 or min_duration_ms <= 0:
        return _merge_consecutive(segments)

    cur = _merge_consecutive(segments)
    while len(cur) > 1:
        candidates = [
            i
            for i, seg in enumerate(cur)
            if (float(seg["end_ms"]) - float(seg["start_ms"])) < min_duration_ms
        ]
        if not candidates:
            break

        idx = min(candidates, key=lambda i: float(cur[i]["end_ms"]) - float(cur[i]["start_ms"]))
        seg = cur[idx]

        left = cur[idx - 1] if idx > 0 else None
        right = cur[idx + 1] if idx + 1 < len(cur) else None

        target: dict[str, Any] | None = None
        if left is not None and right is not None:
            left_active = not left.get("silent", False) and 0 <= left["root_id"] < 12
            right_active = not right.get("silent", False) and 0 <= right["root_id"] < 12
            if left_active and not right_active:
                target = left
            elif right_active and not left_active:
                target = right
            else:
                same_l = 0 <= seg["root_id"] < 12 and left["root_id"] == seg["root_id"]
                same_r = 0 <= seg["root_id"] < 12 and right["root_id"] == seg["root_id"]
                if same_l and not same_r:
                    target = left
                elif same_r and not same_l:
                    target = right
                else:
                    l_score = (float(left["end_ms"]) - float(left["start_ms"])) * max(float(left["confidence"]), 0.05)
                    r_score = (float(right["end_ms"]) - float(right["start_ms"])) * max(float(right["confidence"]), 0.05)
                    target = left if l_score >= r_score else right
        elif left is not None:
            target = left
        else:
            target = right

        if target is None:
            break

        cur[idx] = {
            "start_ms": seg["start_ms"],
            "end_ms": seg["end_ms"],
            "root_id": target["root_id"],
            "quality_id": target["quality_id"],
            "ext_id": target["ext_id"],
            "bass_id": target.get("bass_id", 12),
            "chord": target["chord"],
            "confidence": target["confidence"],
            "silent": target["silent"],
        }
        cur = _merge_consecutive(cur)

    return cur


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
    if getattr(predictor, "is_v5", False) is True:
        return _infer_v5_full_song(
            predictor,
            audio,
            sample_rate=sample_rate,
            crop_s=crop_s,
            step_s=step_s,
            rms_threshold_db=rms_threshold_db,
            confidence_threshold=confidence_threshold,
            min_duration_ms=min_duration_ms,
        )

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

        # Frame-level silence gate so quiet intros inside loud 5s windows are silenced cleanly
        frame_rms_db = _compute_song_frame_rms_db(audio, total_frames)
        active_mask = (acc_w > 0) & (frame_rms_db >= rms_threshold_db)
        frame_ms = MERT_HOP_SAMPLES / float(sample_rate) * 1000.0
        onset_weights = _compute_onset_weights(audio, total_frames)

        decoded = np.zeros((total_frames, 5), dtype=np.float32)
        decoded[:, 0:3] = -1
        decoded[:, 3] = 12

        f_idx = 0
        while f_idx < total_frames:
            if not active_mask[f_idx]:
                f_idx += 1
                continue
            f_end = f_idx + 1
            while f_end < total_frames and active_mask[f_end]:
                f_end += 1
            inv_w = 1.0 / acc_w[f_idx:f_end, None]
            decoded[f_idx:f_end] = decode_musical_sequence(
                acc_r[f_idx:f_end] * inv_w,
                acc_q[f_idx:f_end] * inv_w,
                acc_e[f_idx:f_end] * inv_w,
                acc_b[f_idx:f_end] * inv_w,
                combo_mask=predictor.combo_mask,
                onset_weights=onset_weights[f_idx:f_end],
                frame_ms=frame_ms,
            )
            f_idx = f_end

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

        merged = _prune_short_segments(frame_results, min_duration_ms=min_duration_ms)
        for seg in merged:
            if not seg["silent"] and seg["confidence"] < confidence_threshold:
                seg["root_id"] = -1
                seg["quality_id"] = -1
                seg["ext_id"] = -1
                seg["bass_id"] = 12
                seg["chord"] = "N"
        merged = _merge_consecutive(merged)
        if AUTO_KEY_SPELLING:
            merged = apply_key_spelling(merged)
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
                r, q, e, b = canonicalize_chord_tuple(int(r), int(q), int(e), int(b))
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

    merged = _prune_short_segments(frame_results, min_duration_ms=min_duration_ms)
    if AUTO_KEY_SPELLING:
        merged = apply_key_spelling(merged)
    if merged:
        merged[0]["start_ms"] = 0.0
        merged[-1]["end_ms"] = duration_ms
    return merged


def _filter_no_chord_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Omit silent / `N` segments from user-facing output while keeping exact start/end timestamps of real chords."""
    return [
        s for s in segments
        if not s.get("silent", False)
        and 0 <= int(s.get("root_id", -1)) < 12
        and s.get("chord", "N") != "N"
    ]


def infer_audio(
    predictor: ONNXPredictor,
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    crop_s: int = CROP_S,
    step_s: float = STEP_S,
    rms_threshold_db: float = RMS_THRESHOLD_DB,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    min_duration_ms: float = MIN_DURATION_MS,
    include_silence: bool = False,
) -> list[dict[str, Any]]:
    if getattr(predictor, "is_frame_level", False):
        segments = _infer_frame_level(
            predictor,
            audio,
            sample_rate=sample_rate,
            crop_s=crop_s,
            step_s=step_s,
            rms_threshold_db=rms_threshold_db,
            confidence_threshold=confidence_threshold,
            min_duration_ms=min_duration_ms,
        )
        return segments if include_silence else _filter_no_chord_segments(segments)

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

    segments = _vote_and_merge(raw_preds, audio_len, sample_rate, step_samples, crop_samples)
    return segments if include_silence else _filter_no_chord_segments(segments)


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
        b_id = canonical_bass_id(pred["root_id"], pred.get("bass_id", 12))
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
    return canonicalize_chord_tuple(
        int(r["root_id"]),
        int(r["quality_id"]),
        int(r["ext_id"]),
        int(r.get("bass_id", 12)),
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

        if run_len <= max_jitter_frames and run_end + 1 < n and _chord_key(out[i]) != _chord_key(out[i - 1]):
            if _chord_key(out[i - 1]) == _chord_key(out[run_end + 1]):
                ref = out[i - 1]
            else:
                ref = out[i - 1] if out[i - 1].get("confidence", 0.0) >= out[run_end + 1].get("confidence", 0.0) else out[run_end + 1]
            for k in range(i, run_end + 1):
                out[k]["root_id"] = ref["root_id"]
                out[k]["quality_id"] = ref["quality_id"]
                out[k]["ext_id"] = ref["ext_id"]
                out[k]["bass_id"] = canonical_bass_id(ref["root_id"], ref.get("bass_id", 12))
                out[k]["chord"] = ref["chord"]
                out[k]["silent"] = ref["silent"]
        i = run_end + 1

    return out


def _merge_consecutive(results: list[dict]) -> list[dict]:
    if not results:
        return results
    first = results[0].copy()
    r0, q0, e0, b0 = _chord_key(first)
    first["root_id"], first["quality_id"], first["ext_id"], first["bass_id"] = r0, q0, e0, b0
    if r0 >= 0 and first.get("chord") in (None, "", "N"):
        first["chord"] = chord_name(r0, q0, e0, b0)
    merged = [first]

    for r in results[1:]:
        prev = merged[-1]
        rc, qc, ec, bc = _chord_key(r)
        same = (rc, qc, ec, bc) == (prev["root_id"], prev["quality_id"], prev["ext_id"], prev.get("bass_id", 12))
        if same and bool(r.get("silent", False)) == bool(prev.get("silent", False)):
            w_prev = max(float(prev["end_ms"]) - float(prev["start_ms"]), 1.0)
            w_cur = max(float(r["end_ms"]) - float(r["start_ms"]), 1.0)
            prev["confidence"] = round(
                (float(prev["confidence"]) * w_prev + float(r["confidence"]) * w_cur) / (w_prev + w_cur),
                4,
            )
            prev["end_ms"] = r["end_ms"]
        else:
            item = r.copy()
            item["root_id"], item["quality_id"], item["ext_id"], item["bass_id"] = rc, qc, ec, bc
            merged.append(item)
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

    app = FastAPI(title="ChordSense Pro — Chord Recognition API", version="1.2.0")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": MODEL_PATH,
            "arch_version": 5 if (_predictor is not None and _predictor.is_v5) else 4,
            "demucs_model": DEMUCS_MODEL,
            "demucs_mode": DEMUCS_MODE_DEFAULT,
        }

    @app.post("/predict")
    async def predict(
        file: UploadFile = File(...),
        step_s: float = Query(default=STEP_S, ge=0.1, le=5.0),
        rms_threshold_db: float = Query(default=RMS_THRESHOLD_DB),
        confidence_threshold: float = Query(default=CONFIDENCE_THRESHOLD, ge=0.0, le=1.0),
        separate: bool = Query(default=USE_DEMUCS_DEFAULT, description="Strip vocals & drums via htdemucs_ft"),
        demucs_mode: str = Query(default=DEMUCS_MODE_DEFAULT, description="'auto' (bypass solo piano/guitar) | 'always' | 'off'"),
        include_silence: bool = Query(default=False, description="Include N (silent) segments in output"),
    ):
        if not (file.filename or "").lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
            raise HTTPException(status_code=400, detail="Unsupported file format")

        data = await file.read()
        t_start = time.perf_counter()
        try:
            sep = get_separator() if (separate and demucs_mode.lower() != "off") else None
            audio = load_audio_bytes(
                data,
                SAMPLE_RATE,
                file.filename or "",
                separate=separate,
                separator=sep,
                demucs_mode=demucs_mode,
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
            include_silence=include_silence,
        )
        elapsed_ms = round((time.perf_counter() - t_start) * 1000, 1)

        return JSONResponse({
            "duration_ms": round(len(audio) / SAMPLE_RATE * 1000, 1),
            "separated": separate and demucs_mode.lower() != "off",
            "demucs_mode": demucs_mode,
            "segments": len(segments),
            "elapsed_ms": elapsed_ms,
            "chords": segments,
        })

    @app.post("/predict_npy")
    async def predict_npy(
        file: UploadFile = File(...),
        step_s: float = Query(default=STEP_S, ge=0.1, le=5.0),
        include_silence: bool = Query(default=False),
    ):
        data = await file.read()
        try:
            audio = np.frombuffer(data, dtype=np.int16)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"NPY decode failed: {exc}")

        predictor = get_predictor()
        segments = infer_audio(predictor, audio, sample_rate=SAMPLE_RATE, step_s=step_s, include_silence=include_silence)
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
