from __future__ import annotations
from dataclasses import dataclass


@dataclass
class TrainingSection:
    song_id:         str
    source_type:     str
    source_url:      str
    audio_path:      str
    raw_audio_path:  str | None
    duration_ms:     int
    align_offset_ms: int
    align_score:     float | None
    align_status:    str
    split:           str
    pipeline_version: int = 2


@dataclass
class TrainingChord:
    song_id:      str
    start_ms:     int
    end_ms:       int
    label_raw:    str
    label_status: str
    root_id:      int | None
    quality_id:   int | None
    ext_id:       int | None

