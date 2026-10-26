from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from src.analysis_contract.models import (
    ErrorCode,
    JobError,
    JobProgress,
    JobStage,
    JobState,
    SourceKind,
)
from src.ingestion.domain.errors import InvalidTransition
from src.ingestion.domain.job import Job, advance, new_job, to_progress

T0 = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)
LATER = T0 + timedelta(seconds=5)
ERROR = JobError(code=ErrorCode.AI_FAILURE, message="boom", retryable=True)


def queued() -> Job:
    return new_job(owner_id=uuid4(), source_kind=SourceKind.FILE, title="Song", now=T0)


def processing(stage=JobStage.NORMALIZING, progress=0.15) -> Job:
    return advance(queued(), state=JobState.PROCESSING, stage=stage, progress=progress, now=LATER)


class TestNewJob:
    def test_starts_queued_with_no_stage_and_zero_progress(self):
        job = queued()
        assert (job.state, job.stage, job.progress) == (JobState.QUEUED, None, 0.0)
        assert job.quota_charged is True
        assert job.created_at == job.updated_at == T0

    def test_progress_view_passes_the_contract_validator(self):
        view = to_progress(queued())
        assert isinstance(view, JobProgress)
        assert view.state == JobState.QUEUED


class TestValidTransitions:
    def test_queued_to_processing_needs_a_stage(self):
        job = processing()
        assert (job.state, job.stage, job.progress, job.updated_at) == (
            JobState.PROCESSING,
            JobStage.NORMALIZING,
            0.15,
            LATER,
        )
        with pytest.raises(InvalidTransition):
            advance(queued(), state=JobState.PROCESSING, now=LATER)

    def test_processing_moves_forward_in_stage_and_progress(self):
        job = advance(
            processing(),
            state=JobState.PROCESSING,
            stage=JobStage.SEPARATING,
            progress=0.45,
            now=LATER,
        )
        assert (job.stage, job.progress) == (JobStage.SEPARATING, 0.45)

    def test_processing_may_report_more_progress_inside_one_stage(self):
        job = advance(
            processing(),
            state=JobState.PROCESSING,
            stage=JobStage.NORMALIZING,
            progress=0.2,
            now=LATER,
        )
        assert job.progress == 0.2

    def test_done_needs_a_chart_and_forces_progress_one(self):
        chart_id = uuid4()
        job = advance(processing(), state=JobState.DONE, chart_id=chart_id, now=LATER)
        assert (job.state, job.progress, job.result_chart_id) == (JobState.DONE, 1.0, chart_id)
        assert job.stage == JobStage.NORMALIZING  # last stage reached is kept

    def test_cache_hit_may_finish_straight_from_queued(self):
        job = advance(queued(), state=JobState.DONE, chart_id=uuid4(), now=LATER)
        assert (job.state, job.stage, job.progress) == (JobState.DONE, None, 1.0)

    @pytest.mark.parametrize("start", [queued, processing])
    def test_failed_needs_an_error_from_queued_or_processing(self, start):
        job = advance(start(), state=JobState.FAILED, error=ERROR, now=LATER)
        assert (job.state, job.error) == (JobState.FAILED, ERROR)
        assert to_progress(job).error == ERROR


class TestInvalidTransitions:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"state": JobState.QUEUED},
            {"state": JobState.PROCESSING, "stage": JobStage.NORMALIZING, "progress": 0.1},
            {"state": JobState.DONE, "chart_id": uuid4()},
            {"state": JobState.FAILED, "error": ERROR},
        ],
    )
    def test_terminal_states_never_move_again(self, kwargs):
        for terminal in (
            advance(processing(), state=JobState.DONE, chart_id=uuid4(), now=LATER),
            advance(processing(), state=JobState.FAILED, error=ERROR, now=LATER),
        ):
            with pytest.raises(InvalidTransition):
                advance(terminal, now=LATER, **kwargs)

    def test_progress_never_decreases(self):
        with pytest.raises(InvalidTransition):
            advance(
                processing(progress=0.5),
                state=JobState.PROCESSING,
                stage=JobStage.NORMALIZING,
                progress=0.4,
                now=LATER,
            )

    def test_stage_never_goes_back(self):
        job = processing(stage=JobStage.SEPARATING, progress=0.45)
        with pytest.raises(InvalidTransition):
            advance(
                job, state=JobState.PROCESSING, stage=JobStage.FETCHING, progress=0.5, now=LATER
            )

    @pytest.mark.parametrize("progress", [-0.1, 1.01])
    def test_progress_must_be_a_fraction(self, progress):
        with pytest.raises(InvalidTransition):
            advance(
                queued(),
                state=JobState.PROCESSING,
                stage=JobStage.NORMALIZING,
                progress=progress,
                now=LATER,
            )

    def test_done_without_a_chart_and_failed_without_an_error_are_refused(self):
        with pytest.raises(InvalidTransition):
            advance(processing(), state=JobState.DONE, now=LATER)
        with pytest.raises(InvalidTransition):
            advance(processing(), state=JobState.FAILED, now=LATER)

    def test_queued_cannot_requeue(self):
        with pytest.raises(InvalidTransition):
            advance(queued(), state=JobState.QUEUED, now=LATER)
