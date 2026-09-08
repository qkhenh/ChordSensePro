"""data_processing entrypoint — single public function called by data_ingest."""
from __future__ import annotations

from src.data_loader.domain.models.download_response import DownloadResponse
from src.data_processing.domain.models.processed_audio import ProcessedAudio
from src.data_processing.application.pipeline_factory import build_pipeline


def run_data_processing(response: DownloadResponse) -> ProcessedAudio:
    """Entry point for the T (Transform) phase.

    Builds the handler chain and runs it against the downloaded audio.
    Called exclusively by data_ingest.IngestService — not by DAGs directly.

    Args:
        response: DownloadResponse from run_data_loader() with wav_path set.

    Returns:
        ProcessedAudio — data bag enriched by all 4 handlers.
        Check processed.is_failed before using segments.
    """
    data = ProcessedAudio(
        audio_file=response.audio_file,
        wav_path=response.wav_path,
    )
    pipeline = build_pipeline()
    return pipeline.handle(data)
