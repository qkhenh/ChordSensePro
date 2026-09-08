"""pipeline_factory — wires handler chains for inference and training pipelines."""
from __future__ import annotations

from src.data_processing.application.pipeline_handlers.separate_handler import SeparateHandler
from src.data_processing.application.pipeline_handlers.beat_handler import BeatHandler
from src.data_processing.application.pipeline_handlers.segment_handler import SegmentHandler
from src.data_processing.application.pipeline_handlers.feature_handler import FeatureHandler
from src.data_processing.application.pipeline_handlers.annotation_handler import AnnotationHandler
from src.data_processing.application.pipeline_handlers.augment_handler import AugmentHandler
from src.data_processing.domain.models.base_handler import BaseProcessingHandler


def build_pipeline() -> BaseProcessingHandler:
    """Build inference pipeline: Separate → Beat → Segment → Feature.

    Used for Luồng 1 (user submits song via BE) and Luồng 3 (analytics).
    No augmentation, no annotation parsing.

    Returns:
        Head of the chain (SeparateHandler).
    """
    separate = SeparateHandler()
    beat     = BeatHandler()
    segment  = SegmentHandler()
    feature  = FeatureHandler()

    separate.set_next(beat).set_next(segment).set_next(feature)
    return separate


def build_training_pipeline() -> BaseProcessingHandler:
    """Build training pipeline: Separate → Beat → Segment → Annotation → Augment → Feature.

    Used for Luồng 2 (dag_dataset_ingest — training data ETL).
    Requires ProcessedAudio.annotation_path to be set before calling handle().

    Augmentation strategy (label-aware):
      Basic chords:         ×12 keys only
      Semi-extended (7th):  ×12 keys + noise variants  → 2× oversample
      Extended (9th+, alt): ×12 keys + noise + reverb → 4× oversample

    Returns:
        Head of the chain (SeparateHandler).
    """
    separate   = SeparateHandler()
    beat       = BeatHandler()
    segment    = SegmentHandler()
    annotation = AnnotationHandler()
    augment    = AugmentHandler()
    feature    = FeatureHandler()

    separate.set_next(beat).set_next(segment).set_next(annotation).set_next(augment).set_next(feature)
    return separate
