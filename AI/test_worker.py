"""
ChordSensePro — Background Worker
==================================
Chạy inference (Demucs + ONNX) trong thread riêng, tránh block Flask server.
Mỗi job có id riêng; Flask poll queue để lấy progress/result.
"""
from __future__ import annotations

import gc
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------

@dataclass
class Job:
    id: str
    filename: str
    # Queue of progress dicts: {"pct": int, "msg": str} or {"done": True, "result": ...} or {"error": str}
    q: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=64))
    thread: threading.Thread | None = None
    started_at: float = field(default_factory=time.perf_counter)


# Global job registry (keeps last N jobs)
_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()
MAX_JOBS = 20


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


# ---------------------------------------------------------------------------
# Worker function (runs in a background thread)
# ---------------------------------------------------------------------------

def run_inference_job(
    job: Job,
    filepath: Path,
    use_demucs: bool,
    predictor: Any,
    separator_ref: list,  # mutable container so we can store/update singleton
    separator_lock: threading.Lock,
) -> None:
    """
    Run full inference pipeline in a background thread.
    Posts progress dicts to job.q; last item has {"done": True, ...}.
    """
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
            pass  # drop if consumer is slow

    try:
        t0 = time.perf_counter()

        push(5, "📂 Đang đọc file audio...")
        with open(filepath, "rb") as f:
            raw_bytes = f.read()

        demucs_mode = "auto" if use_demucs else "off"
        need_sep = False

        if demucs_mode != "off":
            push(10, "🔍 Phát hiện loại nhạc cụ...")
            wav_preview, sr_preview = _decode_raw_waveform(raw_bytes, filename=filepath.name)
            if detect_solo_clean_audio(wav_preview, sr_preview):
                need_sep = False
                push(15, "✅ Nhạc cụ solo → bỏ qua Demucs")
            else:
                need_sep = True
                push(15, "🎵 Phát hiện nhiều nhạc cụ → tách bằng Demucs...")
            del wav_preview

        sep = None
        if need_sep:
            # Serialize Demucs loading/usage — it's not thread-safe
            with separator_lock:
                if separator_ref[0] is None:
                    push(20, "⬇️  Tải Demucs model (htdemucs_ft)...")
                    separator_ref[0] = HarmonicSeparator(model_name="htdemucs_ft", segment=6.0)
                sep = separator_ref[0]
                push(30, "🎛️  Demucs đang tách vocal/drums...")
                audio = load_audio_bytes(
                    raw_bytes,
                    target_sr=24000,
                    filename=filepath.name,
                    separate=True,
                    separator=sep,
                    demucs_mode="always",
                )
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

        push(60, "🤖 AI đang nhận diện hợp âm (ONNX)...")
        results = infer_audio(predictor, audio, step_s=2.5, include_silence=False)

        push(90, "📊 Đang tạo waveform peaks...")
        peaks: list[float] = []
        try:
            arr = audio.astype(float) / 32768.0
            n_peaks = 1000
            bucket = max(1, len(arr) // n_peaks)
            for i in range(n_peaks):
                chunk = arr[i * bucket: i * bucket + bucket]
                peaks.append(float(max(abs(chunk.max()), abs(chunk.min()))) if len(chunk) else 0.0)
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
