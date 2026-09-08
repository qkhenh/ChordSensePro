"""ProcessedAudio — data bag that flows through the entire handler chain."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.data_loader.domain.models.audio_file import AudioFile

from src.data_processing.domain.models.audio_segment import AudioSegment


@dataclass
class ProcessedAudio:
    """Data bag passed through SeparateHandler → BeatHandler → SegmentHandler → FeatureHandler.

    Each handler reads its required inputs and writes its outputs into this object.
    If error is set, subsequent handlers short-circuit.
    """
    audio_file:  AudioFile

    # Populated by each handler in sequence
    wav_path:    Path | None = None          # set by data_loader (input to chain)
    stem_path:   Path | None = None          # set by SeparateHandler
    beat_times:  list[float] = field(default_factory=list)  # set by BeatHandler
    tempo_bpm:   float | None = None         # set by BeatHandler
    segments:    list[AudioSegment] = field(default_factory=list)  # set by FeatureHandler

    error:       str | None = None           # set on failure, causes downstream short-circuit

    @property
    def is_failed(self) -> bool:
        return self.error is not None
