"""RawAudioJob — domain model for the staging layer between download and processing.

Stored in MongoDB (collection: raw_audio_jobs).
Represents one downloaded WAV file waiting to be processed by the pipeline.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class RawJobStatus(str, Enum):
    PENDING    = "pending_processing"   # downloaded, waiting for DAG 2
    PROCESSING = "processing"           # DAG 2 picked it up
    DONE       = "done"                 # processed + saved to PostgreSQL
    FAILED     = "failed"               # processing failed — see error_message


class RawIngestOutcome(str, Enum):
    """Result of IngestRawService.run() — replaces ambiguous (None, None) tuple."""
    STAGED  = "staged"    # downloaded WAV, saved to MongoDB
    SKIPPED = "skipped"   # already exists in MongoDB — dedup
    FAILED  = "failed"    # download error


@dataclass
class RawIngestResult:
    """Typed result from IngestRawService.run()."""
    outcome:  RawIngestOutcome
    job:      "RawAudioJob | None" = None
    error:    str | None           = None

    @classmethod
    def staged(cls, job: "RawAudioJob") -> "RawIngestResult":
        return cls(outcome=RawIngestOutcome.STAGED, job=job)

    @classmethod
    def skipped(cls) -> "RawIngestResult":
        return cls(outcome=RawIngestOutcome.SKIPPED)

    @classmethod
    def failed(cls, error: str) -> "RawIngestResult":
        return cls(outcome=RawIngestOutcome.FAILED, error=error)


@dataclass
class RawAudioJob:
    """One downloaded audio file waiting to be processed.

    Created by IngestRawService (DAG 1).
    Consumed by ProcessAudioService (DAG 2).
    """
    id:               str
    source_url:       str
    source_type:      str           # AudioSource value: "choco", "mcgill", "jaah", etc.
    wav_path:         str           # absolute path to WAV on disk: /data/tmp/{id}/song.wav
    annotation_path:  str = ""     # absolute path to annotation file (.jams/.json); "" if not available
    crawl_queue_id:   str = ""     # FK to crawl_queue.id — DAG 2 uses this to update status
    status:           RawJobStatus  = RawJobStatus.PENDING
    error_message:    str | None    = None
    created_at:       datetime      = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at:       datetime      = field(default_factory=lambda: datetime.now(timezone.utc))

    def mark_processing(self) -> None:
        self.status     = RawJobStatus.PROCESSING
        self.updated_at = datetime.now(timezone.utc)

    def mark_done(self) -> None:
        self.status     = RawJobStatus.DONE
        self.updated_at = datetime.now(timezone.utc)

    def mark_failed(self, error: str) -> None:
        self.status        = RawJobStatus.FAILED
        self.error_message = error
        self.updated_at    = datetime.now(timezone.utc)
