"""AnnotationHandler — assigns chord_label to each AudioSegment from annotation file.

Position in training pipeline:
    SeparateHandler → BeatHandler → SegmentHandler → [AnnotationHandler] → AugmentHandler → FeatureHandler

Reads ProcessedAudio.annotation_path, parses chord timeline, then for each segment
finds the chord covering its midpoint and assigns it as segment.chord_label.

Skipped (pass-through) if annotation_path is None — safe for inference pipeline.
"""
from __future__ import annotations

from src.data_processing.domain.models.base_handler import BaseProcessingHandler
from src.data_processing.domain.models.processed_audio import ProcessedAudio
from src.data_processing.application.annotation_parsers.parsers import get_parser


class AnnotationHandler(BaseProcessingHandler):
    """Assign chord_label to each segment using the dataset annotation file.

    Uses the segment's midpoint (ms) to find the matching chord event.
    Segments with no matching chord get label 'N' (no chord / silence).
    """

    def handle(self, data: ProcessedAudio) -> ProcessedAudio:
        if data.is_failed:
            return data

        # No annotation file → inference pipeline, skip silently
        if data.annotation_path is None:
            return self._call_next(data)

        if not data.annotation_path.exists():
            data.error = f"Annotation file not found: {data.annotation_path}"
            return data

        try:
            parser   = get_parser(data.annotation_path)
            timeline = parser.parse(data.annotation_path)
        except Exception as e:
            data.error = f"Annotation parse failed: {e}"
            return data

        # Assign chord_label to each segment by midpoint lookup
        for segment in data.segments:
            mid_sec = (segment.start_ms + segment.end_ms) / 2 / 1000.0
            segment.chord_label = self._find_chord(mid_sec, timeline)

        return self._call_next(data)

    @staticmethod
    def _find_chord(mid_sec: float, timeline: list[tuple[float, float, str]]) -> str:
        """Find the chord label covering mid_sec. Returns 'N' if no match."""
        for start, end, label in timeline:
            if start <= mid_sec < end:
                return label
        return "N"
