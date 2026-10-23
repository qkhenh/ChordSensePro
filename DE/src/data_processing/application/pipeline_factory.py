"""pipeline_factory — wires handler chains for inference and training pipelines."""
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
    """Build inference pipeline: Separate → Beat → Segment → Feature.

    Used for Luồng 1 (user submits song via BE) and Luồng 3 (analytics).
    No augmentation, no annotation parsing.

    Returns:
        Head of the chain (SeparateHandler).
    """
    separate = SeparateHandler()
    beat     = BeatHandler()
    segment  = SegmentHandler()
    summary  = SummaryHandler()
    feature  = FeatureHandler()

    separate.set_next(beat).set_next(segment).set_next(summary).set_next(feature)
    return separate


def build_training_pipeline() -> BaseProcessingHandler:
    """Build training pipeline:
    AlignmentCheck → Separate → Beat → Segment → Annotation → Label → SampleWriter.

    Used for Luồng 2 (dag_process_audio — training data ETL).
    Requires ProcessedAudio.annotation_path for annotated sources (JAAH/ChoCo/...);
    Kaggle derives its label from the filename.

    Output = clean labeled 2s segments (24 kHz int16 .npy + rows in ``training_samples``).
    AugmentHandler / FeatureHandler / SummaryHandler are intentionally NOT in this chain:
    augmentation is applied on the train split only, at training time, and features are
    computed by the model. Those handlers remain for the inference pipeline / future use.

    Returns:
        Head of the chain (AlignmentCheckHandler).
    """
    align      = AlignmentCheckHandler()
    separate   = SeparateHandler()
    beat       = BeatHandler()
    segment    = SegmentHandler()
    annotation = AnnotationHandler()
    label      = LabelHandler()
    writer     = SampleWriterHandler()

    align.set_next(separate).set_next(beat).set_next(segment).set_next(annotation).set_next(label).set_next(writer)
    return align
