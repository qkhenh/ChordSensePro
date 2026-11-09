from __future__ import annotations
import io
import os
import shutil
from pathlib import Path

from src.data_processing.domain.models.base_handler import BaseProcessingHandler
from src.data_processing.domain.models.processed_audio import ProcessedAudio
from src.shared.domain.dataset_split import split_for

TARGET_SR      = 24_000
SEGMENT_SEC    = 2
SEGMENT_LEN    = TARGET_SR * SEGMENT_SEC
USABLE_STATUS  = ("ok", "partial")


def training_dir() -> Path:
    return Path(os.getenv("TRAINING_DIR") or (Path(os.getenv("DATA_DIR", "/data")) / "training"))


def training_v2_dir() -> Path:
    return Path(os.getenv("TRAINING_V2_DIR") or (Path(os.getenv("DATA_DIR", "/data")) / "training_v2"))


def raw_audio_dir() -> Path:
    return Path(os.getenv("RAW_AUDIO_DIR") or (Path(os.getenv("DATA_DIR", "/data")) / "raw"))


class SampleWriterHandler(BaseProcessingHandler):

    def handle(self, data: ProcessedAudio) -> ProcessedAudio:
        if data.is_failed:
            return data

        if data.chord_timeline:
            return self._handle_v2(data)
        return self._handle_v1(data)

    def _handle_v2(self, data: ProcessedAudio) -> ProcessedAudio:
        if not data.stem_path or not data.stem_path.exists():
            data.error = "SampleWriter v2: stem_path missing"
            return data

        usable = [c for c in data.chord_timeline if c.get("label_status") in USABLE_STATUS]
        if not usable:
            data.error = (
                f"No usable labeled chords (total={len(data.chord_timeline)}) "
                f"— annotation missing or labels unmapped"
            )
            return data

        try:
            import numpy as np
            import soundfile as sf
            import librosa

            audio, file_sr = sf.read(str(data.stem_path), dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if file_sr != TARGET_SR:
                audio = librosa.resample(audio, orig_sr=file_sr, target_sr=TARGET_SR)
            audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)

            song_id = data.audio_file.analysis_id
            rel_path = f"audio/{song_id}.npy"
            abs_path = training_v2_dir() / rel_path
            self._atomic_save(abs_path, audio_int16)

            raw_dest = None
            if data.wav_path and data.wav_path.exists():
                raw_dest_path = raw_audio_dir() / f"{song_id}.wav"
                raw_dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(data.wav_path), str(raw_dest_path))
                raw_dest = f"raw/{song_id}.wav"

            duration_ms = round(len(audio) / TARGET_SR * 1000)
            split = split_for(data.audio_file.source_url)
            source_type = getattr(data.audio_file.source_type, "value", str(data.audio_file.source_type))

            data.section_row = {
                "song_id":         song_id,
                "source_type":     source_type,
                "source_url":      data.audio_file.source_url,
                "audio_path":      rel_path,
                "raw_audio_path":  raw_dest,
                "duration_ms":     duration_ms,
                "align_offset_ms": data.align_offset_ms,
                "align_score":     data.align_score,
                "align_status":    data.align_status,
                "split":           split,
                "pipeline_version": 2,
            }

            data.chord_rows = [
                {
                    "song_id":      song_id,
                    "start_ms":     c["start_ms"],
                    "end_ms":       c["end_ms"],
                    "label_raw":    c["label_raw"],
                    "label_status": c["label_status"],
                    "root_id":      c.get("root_id"),
                    "quality_id":   c.get("quality_id"),
                    "ext_id":       c.get("ext_id"),
                }
                for c in data.chord_timeline
            ]

        except Exception as e:
            data.error = f"SampleWriter v2 failed: {e}"
            return data

        return self._call_next(data)

    def _handle_v1(self, data: ProcessedAudio) -> ProcessedAudio:
        labeled = [s for s in data.segments if s.chord_label is not None and s.label_status]
        usable  = [s for s in labeled if s.label_status in USABLE_STATUS]
        if not usable:
            data.error = (
                f"No usable labeled segments (segments={len(data.segments)}, "
                f"labeled={len(labeled)}) — annotation missing or labels unmapped"
            )
            return data

        try:
            import numpy as np
            clips = [self._to_int16_clip(s.audio_data, s.sample_rate) for s in labeled]
            array = np.stack(clips).astype(np.int16)

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
        import numpy as np
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".npy.tmp")
        with open(tmp, "wb") as f:
            np.save(f, array)
        os.replace(tmp, path)
