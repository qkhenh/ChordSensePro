"""AlignmentCheckHandler — rejects JAAH tracks whose YouTube audio does not match the annotation.

JAAH annotations were made on a specific CD rip; the crawler fetches *some* YouTube upload by
``artist + title`` search. A different edit/live take shifts every chord in time and silently
poisons the labels. The cheapest reliable guard is the total duration: JAAH JSON stores the exact
duration of the annotated recording, so we compare it with the downloaded WAV.

Position in training pipeline (first handler):
    [AlignmentCheck] → Separate → Beat → Segment → Annotation → Label → SampleWriter

Enforced for JAAH and ChoCo (total duration) and Hooktheory (WAV length vs requested section).
Other sources set align_status = "n/a" (Kaggle = isolated studio chords;
ChoCo audio partitions are checked the same way (JAMS file_metadata.duration).
"""
from __future__ import annotations
import os

from src.data_processing.application.annotation_parsers.parsers import (
    read_annotation_duration, read_annotation_section,
)
from src.data_processing.domain.models.base_handler import BaseProcessingHandler
from src.data_processing.domain.models.processed_audio import ProcessedAudio

DEFAULT_TOLERANCE_SEC = 3.0

# Sources whose audio comes from a YouTube search / link, so a wrong edit is possible.
# ChoCo JAMS store the annotated duration in file_metadata.duration (JAAH JSON: top-level).
DURATION_CHECKED = ("jaah", "choco")

# Hooktheory downloads only a short section: its video-level duration check happens in the downloader,
# here the WAV is compared with the requested section length (catches truncated / wrong-range downloads).
SECTION_CHECKED = "hooktheory"


class AlignmentCheckHandler(BaseProcessingHandler):
    """Compare WAV duration with the annotation's declared duration (JAAH / ChoCo)."""

    def __init__(self, tolerance_sec: float | None = None) -> None:
        super().__init__()
        self.tolerance = (
            tolerance_sec if tolerance_sec is not None
            else float(os.getenv("ALIGN_TOLERANCE_SEC", DEFAULT_TOLERANCE_SEC))
        )

    def handle(self, data: ProcessedAudio) -> ProcessedAudio:
        if data.is_failed:
            return data

        # .value string compare — data_processing must not import data_loader at runtime
        source = getattr(data.audio_file.source_type, "value", str(data.audio_file.source_type))
        checked = source in DURATION_CHECKED or source == SECTION_CHECKED
        if not checked or data.annotation_path is None or data.wav_path is None:
            data.align_status = "n/a"
            return self._call_next(data)

        if source == SECTION_CHECKED:
            section = read_annotation_section(data.annotation_path)
            expected = (section[1] - section[0]) if section else None
        else:
            expected = read_annotation_duration(data.annotation_path)
        if expected is None:
            data.align_status = "n/a"
            return self._call_next(data)

        try:
            import soundfile as sf  # lazy — audio extras (Docker only)
            actual = float(sf.info(str(data.wav_path)).duration)
        except Exception as e:
            data.error = f"AlignmentCheck: cannot read WAV duration: {e}"
            return data

        diff = abs(actual - expected)
        if diff > self.tolerance:
            data.align_status = "duration_mismatch"
            data.error = (
                f"duration_mismatch: audio {actual:.1f}s vs annotation {expected:.1f}s "
                f"(Δ{diff:.1f}s > {self.tolerance:.1f}s) — wrong YouTube version"
            )
            return data

        data.align_status = "ok"
        return self._call_next(data)
