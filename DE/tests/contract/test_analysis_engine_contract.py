"""Acceptance suite for every ``AnalysisEngine`` adapter.

The mock runs it today; the real DE/AI pipeline adapter must pass the same tests in M5 —
add its factory to ``ENGINES``.
"""

import asyncio
from collections.abc import Callable
from uuid import uuid4

import pytest

from src.analysis_contract.models import (
    AnalysisRequest,
    AnalysisResult,
    ErrorCode,
    JobProgress,
    JobStage,
    JobState,
    SourceKind,
)
from src.analysis_contract.ports import AnalysisEngine, AnalysisError
from src.ingestion.infrastructure.mock_engine import FIXTURES_DIR, MockAnalysisEngine

ENGINES: dict[str, Callable[[], AnalysisEngine]] = {
    "mock": lambda: MockAnalysisEngine(FIXTURES_DIR),
}
HASH = "ab" * 32


@pytest.fixture(params=ENGINES)
def engine(request) -> AnalysisEngine:
    return ENGINES[request.param]()


def make_request(stem: str = "song", content_hash: str = HASH) -> AnalysisRequest:
    return AnalysisRequest(
        job_id=uuid4(),
        content_hash=content_hash,
        source_kind=SourceKind.FILE,
        audio_path=f"/data/tmp/{stem}.wav",
    )


async def collect(engine: AnalysisEngine, request: AnalysisRequest):
    seen: list[JobProgress] = []

    async def on_progress(progress: JobProgress) -> None:
        seen.append(progress)

    return await engine.analyze(request, on_progress), seen


async def test_returns_a_valid_result_for_the_request(engine):
    request = make_request()
    result, _ = await collect(engine, request)
    assert isinstance(result, AnalysisResult)
    AnalysisResult.model_validate(result.model_dump(mode="json"))
    assert result.job_id == request.job_id
    assert result.content_hash == request.content_hash
    assert result.chord_events
    assert result.metadata.model_version
    assert result.metadata.pipeline_version


async def test_progress_is_processing_only_ordered_and_monotonic(engine):
    request = make_request()
    _, seen = await collect(engine, request)
    assert seen, "an engine must report at least one progress step"
    order = list(JobStage)
    for progress in seen:
        assert progress.job_id == request.job_id
        assert progress.state == JobState.PROCESSING  # done/failed belong to the executor
        assert progress.stage is not None
        assert 0.0 <= progress.progress <= 1.0
    stages = [order.index(p.stage) for p in seen]
    assert stages == sorted(stages)
    values = [p.progress for p in seen]
    assert values == sorted(values)
    assert values[-1] < 1.0


async def test_the_same_audio_gets_the_same_chords(engine):
    first, _ = await collect(engine, make_request())
    second, _ = await collect(engine, make_request())
    assert first.chord_events == second.chord_events
    assert first.metadata == second.metadata


async def test_a_failing_analysis_raises_analysis_error_with_a_contract_code(engine):
    with pytest.raises(AnalysisError) as exc:
        await collect(engine, make_request(stem="fail-ai"))
    assert isinstance(exc.value.code, ErrorCode)
    assert exc.value.code == ErrorCode.AI_FAILURE
    assert exc.value.message


async def test_cancellation_is_not_swallowed(engine):
    blocked = asyncio.Event()

    async def stuck(_: JobProgress) -> None:
        blocked.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(engine.analyze(make_request(), stuck))
    await asyncio.wait_for(blocked.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
