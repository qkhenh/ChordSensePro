"""The ``Job`` aggregate and its state machine.

queued -> processing -> done | failed, plus queued -> done (cache hit) and queued -> failed
(rejected before it started). ``done`` and ``failed`` are final; stage and progress only move
forward.
"""

from dataclasses import dataclass, replace
from datetime import date, datetime
from uuid import UUID, uuid4

from src.analysis_contract.models import (
    JobError,
    JobProgress,
    JobStage,
    JobState,
    SourceKind,
)
from src.ingestion.domain.errors import InvalidTransition

_STAGE_ORDER = list(JobStage)
_FINAL = (JobState.DONE, JobState.FAILED)


@dataclass(frozen=True)
class Job:
    id: UUID
    owner_id: UUID
    source_kind: SourceKind
    title: str
    created_at: datetime
    updated_at: datetime
    source_url: str | None = None
    input_path: str | None = None
    fingerprint: str | None = None  # sha256 of the uploaded bytes (cheap cache key)
    content_hash: str | None = None  # sha256 of the normalised WAV, known after NORMALIZING
    state: JobState = JobState.QUEUED
    stage: JobStage | None = None
    progress: float = 0.0
    error: JobError | None = None
    result_chart_id: UUID | None = None
    quota_charged: bool = True
    charged_on: date | None = None  # UTC day the quota was taken from, for the refund


def new_job(
    *,
    owner_id: UUID,
    source_kind: SourceKind,
    title: str,
    now: datetime,
    source_url: str | None = None,
    input_path: str | None = None,
    fingerprint: str | None = None,
    quota_charged: bool = True,
    charged_on: date | None = None,
    job_id: UUID | None = None,
) -> Job:
    return Job(
        id=job_id or uuid4(),
        owner_id=owner_id,
        source_kind=source_kind,
        title=title,
        created_at=now,
        updated_at=now,
        source_url=source_url,
        input_path=input_path,
        fingerprint=fingerprint,
        quota_charged=quota_charged,
        charged_on=charged_on,
    )


def to_progress(job: Job) -> JobProgress:
    return JobProgress(
        job_id=job.id,
        state=job.state,
        stage=job.stage,
        progress=job.progress,
        error=job.error,
        result_chart_id=job.result_chart_id,
    )


def advance(
    job: Job,
    *,
    state: JobState,
    now: datetime,
    stage: JobStage | None = None,
    progress: float | None = None,
    error: JobError | None = None,
    chart_id: UUID | None = None,
) -> Job:
    """Return the job after a legal transition, else raise ``InvalidTransition``."""
    if job.state in _FINAL:
        raise InvalidTransition(f"a {job.state.value} job cannot change")
    if state == JobState.QUEUED:
        raise InvalidTransition("a job cannot go back to queued")

    if state == JobState.PROCESSING:
        if stage is None:
            raise InvalidTransition("processing needs a stage")
        new_progress = job.progress if progress is None else progress
        if new_progress < job.progress:
            raise InvalidTransition("progress cannot decrease")
        if job.stage is not None and _STAGE_ORDER.index(stage) < _STAGE_ORDER.index(job.stage):
            raise InvalidTransition("stage cannot go back")
        changes: dict = {"stage": stage, "progress": new_progress}
    elif state == JobState.DONE:
        if chart_id is None:
            raise InvalidTransition("done needs the resulting chart")
        changes = {"progress": 1.0, "result_chart_id": chart_id}
    else:
        if error is None:
            raise InvalidTransition("failed needs an error")
        changes = {"error": error}

    candidate = replace(job, state=state, updated_at=now, **changes)
    try:
        to_progress(candidate)  # the contract validator is the final word
    except ValueError as exc:
        raise InvalidTransition(str(exc)) from exc
    return candidate
