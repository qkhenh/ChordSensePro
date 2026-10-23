"""LabelHandler — maps each segment's raw Harte label to the 3 class ids (root / quality / extension).

Position in training pipeline:
    ... → Segment → Annotation → [LabelHandler] → SampleWriter

Pure mapping, no I/O. Segments without a label (chord_label is None) are left untouched and are
dropped later by SampleWriterHandler.
"""
from __future__ import annotations

from src.data_processing.application.mappers.harte_mapper import map_harte
from src.data_processing.domain.models.base_handler import BaseProcessingHandler
from src.data_processing.domain.models.processed_audio import ProcessedAudio


class LabelHandler(BaseProcessingHandler):
    """Fill label_status / root_id / quality_id / ext_id on every labeled AudioSegment."""

    def handle(self, data: ProcessedAudio) -> ProcessedAudio:
        if data.is_failed:
            return data

        for segment in data.segments:
            if segment.chord_label is None:
                continue
            mapped = map_harte(segment.chord_label)
            segment.label_status = mapped.status.value
            segment.root_id      = mapped.root_id
            segment.quality_id   = mapped.quality_id
            segment.ext_id       = mapped.ext_id

        return self._call_next(data)
