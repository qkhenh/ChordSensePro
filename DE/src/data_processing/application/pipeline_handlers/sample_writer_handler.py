"""SampleWriterHandler — final training handler: persists the clean labeled segments to disk.

Position in training pipeline (last handler):
    ... → Annotation → Label → [SampleWriter]

What it writes
--------------
One ``.npy`` per song:  ``{TRAINING_DIR}/audio/{song_id}.npy``
    dtype int16, shape (N, 48000)  = N segments × 2 s @ 24 kHz mono  (MERT input rate)

Segments are stored **clean** (no augmentation, no features): augmentation is a train-side
concern applied later, and features are recomputed by the model. Split (train/val/test) is decided
here from a hash of the source URL so every segment of a song stays in the same split.

No database access: it only fills ``ProcessedAudio.sample_rows`` (one dict per saved segment,
keys identical to ``TrainingSample`` fields). ProcessAudioService persists them to PostgreSQL.
"""
from __future__ import annotations
import io
import os
from pathlib import Path

from src.data_processing.domain.models.base_handler import BaseProcessingHandler
from src.data_processing.domain.models.processed_audio import ProcessedAudio
from src.shared.domain.dataset_split import split_for

TARGET_SR      = 24_000
SEGMENT_SEC    = 2
SEGMENT_LEN    = TARGET_SR * SEGMENT_SEC     # 48000 samples
USABLE_STATUS  = ("ok", "partial")


def training_dir() -> Path:
    """Root of the persisted training audio (Docker volume)."""
    return Path(os.getenv("TRAINING_DIR") or (Path(os.getenv("DATA_DIR", "/data")) / "training"))


class SampleWriterHandler(BaseProcessingHandler):
    """Decode → resample 24 kHz → int16 → stack → write one .npy per song."""

    def handle(self, data: ProcessedAudio) -> ProcessedAudio:
        if data.is_failed:
            return data

        labeled = [s for s in data.segments if s.chord_label is not None and s.label_status]
        usable  = [s for s in labeled if s.label_status in USABLE_STATUS]
        if not usable:
            data.error = (
                f"No usable labeled segments (segments={len(data.segments)}, "
                f"labeled={len(labeled)}) — annotation missing or labels unmapped"
            )
            return data

        try:
            import numpy as np  # lazy — audio extras (Docker only)
            clips = [self._to_int16_clip(s.audio_data, s.sample_rate) for s in labeled]
            array = np.stack(clips).astype(np.int16)           # (N, 48000)

            song_id  = data.audio_file.analysis_id
            rel_path = f"audio/{song_id}.npy"
            abs_path = training_dir() / rel_path
            self._atomic_save(abs_path, array)
        except Exception as e:
            data.error = f"SampleWriter failed: {e}"
            return data

        split = split_for(data.audio_file.source_url)
        source_type = getattr(data.audio_file.source_type, "value", str(data.audio_file.source_type))
        data.sample_rows = [
            {
                "song_id":      song_id,
                "source_type":  source_type,
                "source_url":   data.audio_file.source_url,
                "segment_idx":  seg.segment_idx,
                "start_ms":     round(seg.start_ms),
                "end_ms":       round(seg.end_ms),
                "audio_path":   rel_path,
                "audio_idx":    i,
                "sample_rate":  TARGET_SR,
                "label_raw":    seg.chord_label,
                "label_status": seg.label_status,
                "root_id":      seg.root_id,
                "quality_id":   seg.quality_id,
                "ext_id":       seg.ext_id,
                "split":        split,
                "align_status": data.align_status,
            }
            for i, seg in enumerate(labeled)
        ]
        return self._call_next(data)

    @staticmethod
    def _to_int16_clip(wav_bytes: bytes, sr: int):
        """WAV bytes → mono int16 array of exactly SEGMENT_LEN samples at TARGET_SR."""
        import numpy as np
        import soundfile as sf
        audio, file_sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != TARGET_SR:
            import librosa
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=TARGET_SR)
        if len(audio) < SEGMENT_LEN:
            audio = np.pad(audio, (0, SEGMENT_LEN - len(audio)))
        audio = audio[:SEGMENT_LEN]
        return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)

    @staticmethod
    def _atomic_save(path: Path, array) -> None:
        """Write to a temp file then rename — a crash never leaves a half-written .npy."""
        import numpy as np
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".npy.tmp")
        with open(tmp, "wb") as f:
            np.save(f, array)
        os.replace(tmp, path)
