"""
ChordSensePro — Background Worker
==================================
Chạy inference (Demucs GPU + Disk Stem Cache + ONNX) trong thread riêng, tránh block Flask server.
Tự động quản lý VRAM trên GPU 4GB (tránh xung đột VRAM giữa ONNX Runtime và PyTorch Demucs khiến Demucs bị tụt về CPU).
"""
from __future__ import annotations

import gc
import hashlib
import queue
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------

@dataclass
class Job:
    id: str
    filename: str
    # Queue of progress dicts: {"pct": int, "msg": str} or {"done": True, "result": ...} or {"error": str}
    q: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=128))
    thread: threading.Thread | None = None
    started_at: float = field(default_factory=time.perf_counter)


# Global job registry (keeps last N jobs)
_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()
MAX_JOBS = 20

CACHE_DIR = Path(__file__).resolve().parent / "results" / ".demucs_cache"


def _prune_jobs() -> None:
    with _jobs_lock:
        if len(_jobs) > MAX_JOBS:
            oldest = sorted(_jobs.keys(), key=lambda k: _jobs[k].started_at)
            for k in oldest[:len(_jobs) - MAX_JOBS]:
                del _jobs[k]


def create_job(filename: str) -> Job:
    job = Job(id=str(uuid.uuid4()), filename=filename)
    with _jobs_lock:
        _jobs[job.id] = job
    _prune_jobs()
    return job


def get_job(job_id: str) -> Job | None:
    with _jobs_lock:
        return _jobs.get(job_id)


def _stem_cache_path(filepath: Path, demucs_model: str, target_sr: int = 24000) -> Path:
    st = filepath.stat()
    sig = f"{filepath.name}|{st.st_size}|{st.st_mtime_ns}|{demucs_model}|{target_sr}"
    digest = hashlib.sha1(sig.encode("utf-8")).hexdigest()[:12]
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", filepath.stem)[:40]
    return CACHE_DIR / f"{safe_stem}_{demucs_model}_{digest}.npy"


def _is_low_vram_gpu(threshold_gb: float = 6.5) -> bool:
    try:
        import torch
        if torch.cuda.is_available():
            total_bytes = torch.cuda.get_device_properties(0).total_memory
            return total_bytes < int(threshold_gb * (1024 ** 3))
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Worker function (runs in a background thread)
# ---------------------------------------------------------------------------

def run_inference_job(
    job: Job,
    filepath: Path,
    use_demucs: bool,
    predictor_ref: list,
    predictor_loader: Callable[[], Any],
    separator_dict: dict[str, Any],
    gpu_lock: threading.Lock,
    demucs_model: str = "htdemucs_ft",
) -> None:
    """Run full inference pipeline in a background thread with GPU VRAM swapping & stem caching."""
    from src.serve import (
        HarmonicSeparator,
        _decode_raw_waveform,
        detect_solo_clean_audio,
        infer_audio,
        load_audio_bytes,
    )

    def push(pct: int, msg: str) -> None:
        try:
            job.q.put_nowait({"pct": pct, "msg": msg})
        except queue.Full:
            try:
                job.q.get_nowait()
                job.q.put_nowait({"pct": pct, "msg": msg})
            except Exception:
                pass

    try:
        t0 = time.perf_counter()

        push(5, "📂 Đang đọc file audio...")
        with open(filepath, "rb") as f:
            raw_bytes = f.read()

        need_sep = False
        wav_decoded: np.ndarray | None = None
        sr_decoded: int = 24000

        if use_demucs:
            push(10, "🔍 Kiểm tra nhạc cụ solo / full-band...")
            wav_decoded, sr_decoded = _decode_raw_waveform(raw_bytes, filename=filepath.name)
            if detect_solo_clean_audio(wav_decoded, sr_decoded):
                need_sep = False
                push(16, "✅ Nhạc cụ solo sạch → bỏ qua Demucs")
                del wav_decoded
                wav_decoded = None
            else:
                need_sep = True

        with gpu_lock:
            if need_sep:
                cache_file = _stem_cache_path(filepath, demucs_model, target_sr=24000)
                if cache_file.exists():
                    push(55, f"⚡ Đã có cache Demucs ({demucs_model}) → nạp tức thì!")
                    audio = np.load(cache_file)
                    if wav_decoded is not None:
                        del wav_decoded
                        wav_decoded = None
                else:
                    # Free ONNX CUDA arena on <=6GB GPUs (e.g. RTX 3050 4GB) so Demucs runs 100% on GPU without CPU fallback!
                    if predictor_ref[0] is not None and _is_low_vram_gpu(6.5):
                        push(18, "🧹 Giải phóng VRAM tạm thời cho Demucs chạy full tốc độ GPU...")
                        predictor_ref[0] = None
                        gc.collect()
                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            pass

                    sep = separator_dict.get(demucs_model)
                    if sep is None:
                        push(20, f"⬇️ Khởi tạo Demucs ({demucs_model})...")
                        sep = HarmonicSeparator(model_name=demucs_model, segment=6.0)
                        separator_dict[demucs_model] = sep

                    dev_label = str(getattr(sep, "device", "cuda")).upper()
                    push(22, f"🎛️ Demucs ({demucs_model} · {dev_label}) đang tách: 0%...")
                    last_pct = [-1]

                    def _on_pct(frac: float) -> None:
                        p_int = int(round(frac * 100))
                        if p_int != last_pct[0]:
                            last_pct[0] = p_int
                            ui_bar = 22 + int(round(frac * 46))  # 22% -> 68%
                            push(ui_bar, f"🎛️ Demucs ({demucs_model} · {dev_label}) đang tách: {p_int}%...")

                    if wav_decoded is None:
                        wav_decoded, sr_decoded = _decode_raw_waveform(raw_bytes, filename=filepath.name)

                    audio = sep.separate_waveform(
                        wav_decoded,
                        sr=sr_decoded,
                        target_sr=24000,
                        progress_cb=_on_pct,
                        show_progress=False,
                    )
                    del wav_decoded
                    wav_decoded = None

                    try:
                        CACHE_DIR.mkdir(parents=True, exist_ok=True)
                        np.save(cache_file, audio)
                    except Exception:
                        pass
            else:
                push(45, "🔊 Đang decode và resample audio...")
                audio = load_audio_bytes(
                    raw_bytes,
                    target_sr=24000,
                    filename=filepath.name,
                    separate=False,
                    separator=None,
                    demucs_mode="off",
                )

            del raw_bytes

            if predictor_ref[0] is None:
                push(70, "⚡ Nạp lại ONNX model lên GPU...")
                predictor_ref[0] = predictor_loader()

            push(75, "🤖 AI đang nhận diện hợp âm toàn bài (ONNX GPU)...")
            results = infer_audio(predictor_ref[0], audio, step_s=2.5, include_silence=False)

        push(92, "📊 Đang tạo waveform peaks...")
        peaks: list[float] = []
        try:
            arr = audio.astype(np.float32) / 32768.0
            n_peaks = 1000
            bucket = max(1, len(arr) // n_peaks)
            for i in range(n_peaks):
                chunk = arr[i * bucket: i * bucket + bucket]
                peaks.append(float(np.max(np.abs(chunk))) if len(chunk) else 0.0)
        except Exception:
            peaks = []

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        duration_ms = round(len(audio) / 24000 * 1000, 1)

        job.q.put({
            "done": True,
            "result": {
                "song": filepath.name,
                "duration_ms": duration_ms,
                "elapsed_ms": elapsed_ms,
                "segments": len(results),
                "chords": results,
                "peaks": peaks,
            },
        })

    except Exception as exc:
        job.q.put({"error": str(exc)})
