from __future__ import annotations

from src.data_processing.application.pipeline_handlers.separate_handler import SeparateHandler
from src.data_processing.application.pipeline_handlers.beat_handler import BeatHandler
from src.data_processing.application.pipeline_handlers.segment_handler import SegmentHandler
from src.data_processing.application.pipeline_handlers.feature_handler import FeatureHandler
from src.data_processing.application.pipeline_handlers.annotation_handler import AnnotationHandler
from src.data_processing.application.pipeline_handlers.augment_handler import AugmentHandler
from src.data_processing.application.pipeline_handlers.summary_handler import SummaryHandler
from src.data_processing.application.pipeline_handlers.alignment_check_handler import AlignmentCheckHandler
from src.data_processing.application.pipeline_handlers.label_handler import LabelHandler
from src.data_processing.application.pipeline_handlers.sample_writer_handler import SampleWriterHandler
from src.data_processing.domain.models.base_handler import BaseProcessingHandler


def build_pipeline() -> BaseProcessingHandler:
    separate = SeparateHandler()
    beat     = BeatHandler()
    segment  = SegmentHandler()
    summary  = SummaryHandler()
    feature  = FeatureHandler()

    separate.set_next(beat).set_next(segment).set_next(summary).set_next(feature)
    return separate


def build_training_pipeline() -> BaseProcessingHandler:
    align      = AlignmentCheckHandler()
    separate   = SeparateHandler()
    beat       = BeatHandler()
    segment    = SegmentHandler()
    annotation = AnnotationHandler()
    label      = LabelHandler()
    writer     = SampleWriterHandler()

    align.set_next(separate).set_next(beat).set_next(segment).set_next(annotation).set_next(label).set_next(writer)
    return align


def build_training_pipeline_v2() -> BaseProcessingHandler:
    align      = AlignmentCheckHandler()
    separate   = SeparateHandler()
    annotation = AnnotationHandler()
    label      = LabelHandler()
    writer     = SampleWriterHandler()

    align.set_next(separate).set_next(annotation).set_next(label).set_next(writer)
    return align
