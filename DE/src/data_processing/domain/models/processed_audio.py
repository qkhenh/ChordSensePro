from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.data_loader.domain.models.audio_file import AudioFile

from src.data_processing.domain.models.audio_segment import AudioSegment


@dataclass
class ProcessedAudio:
    audio_file:  AudioFile

    wav_path:        Path | None = None
    stem_path:       Path | None = None
    beat_times:      list[float] = field(default_factory=list)
    tempo_bpm:       float | None = None
    segments:        list[AudioSegment] = field(default_factory=list)
    annotation_path: Path | None = None

    align_status:    str = "n/a"
    align_offset_ms: int = 0
    align_score:     float = 0.0
    sample_rows:     list[dict] = field(default_factory=list)

    chord_timeline:  list[dict] = field(default_factory=list)
    detected_key:    str = ""
    chord_sheet:     dict = field(default_factory=dict)

    section_row:     dict | None = None
    chord_rows:      list[dict] = field(default_factory=list)

    error:       str | None = None

    @property
    def is_failed(self) -> bool:
        return self.error is not None
