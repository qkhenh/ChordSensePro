"""pipeline_factory — wires the handler chain for the Transform phase."""
from __future__ import annotations

from src.data_processing.application.pipeline_handlers.separate_handler import SeparateHandler
from src.data_processing.application.pipeline_handlers.beat_handler import BeatHandler
from src.data_processing.application.pipeline_handlers.segment_handler import SegmentHandler
from src.data_processing.application.pipeline_handlers.feature_handler import FeatureHandler
from src.data_processing.domain.models.base_handler import BaseProcessingHandler


def build_pipeline() -> BaseProcessingHandler:
    """Build the handler chain: Separate → Beat → Segment → Feature.

    Returns the head of the chain (SeparateHandler).
    Caller passes ProcessedAudio to head.handle() to run the full pipeline.
    """
    separate = SeparateHandler()
    beat     = BeatHandler()
    segment  = SegmentHandler()
    feature  = FeatureHandler()

    separate.set_next(beat).set_next(segment).set_next(feature)

    return separate
