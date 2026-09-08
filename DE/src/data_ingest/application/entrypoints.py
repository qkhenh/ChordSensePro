"""data_ingest entrypoint — DAG calls this, nothing else."""
from __future__ import annotations
import asyncio

from src.data_ingest.application.ingest_service import IngestService
from src.data_ingest.domain.models.ingest_record import IngestRecord


def run_ingest_pipeline(
    analysis_id: str,
    source_url: str,
    user_id: str,
    song_title: str = "",
) -> IngestRecord:
    """Entry point for the full ETL pipeline. Called by DAG PythonOperator.

    Bridges async IngestService to synchronous Airflow task context.

    Args:
        analysis_id: Pre-generated UUID for this run
        source_url:  YouTube URL or local file path
        user_id:     Owner of the analysis
        song_title:  Optional title override

    Returns:
        IngestRecord — check record.status and record.error_message.
    """
    service = IngestService()
    return asyncio.run(service.run(
        analysis_id=analysis_id,
        source_url=source_url,
        user_id=user_id,
        song_title=song_title,
    ))
