"""IngestRecord — tracks status of one ingestion run through the pipeline."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum

from src.shared.domain.base_model import BaseEntity


class IngestStatus(str, Enum):
    PENDING    = "pending"
    LOADING    = "loading"      # E: downloading audio
    PROCESSING = "processing"   # T: separating, segmenting, extracting features
    SAVING     = "saving"       # L: writing to PostgreSQL
    DONE       = "done"
    FAILED     = "failed"


@dataclass
class IngestRecord(BaseEntity):
    """Tracks one full ETL run: E → T → L.

    Passed back to the DAG so Airflow can mark task success/failure.
    Not persisted to database — lives in XCom as JSON.
    """
    analysis_id:   str = ""
    status:        IngestStatus = IngestStatus.PENDING
    error_message: str | None = None

    def fail(self, error: str | None) -> None:
        self.status = IngestStatus.FAILED
        self.error_message = error or "Unknown error"
