import asyncio
import json
import time
from uuid import uuid4

import pytest

from src.analysis_contract.models import AnalysisRequest, AnalysisResult, JobStage, SourceKind
from src.analysis_contract.ports import AnalysisError
from src.ingestion.infrastructure.mock_engine import FIXTURES_DIR, MockAnalysisEngine

FIXTURE_NAMES = sorted(p.name.removesuffix(".result.json") for p in FIXTURES_DIR.glob("*.json"))


def request(stem: str, content_hash: str = "ab" * 32) -> AnalysisRequest:
    return AnalysisRequest(
        job_id=uuid4(),
        content_hash=content_hash,
        source_kind=SourceKind.FILE,
        audio_path=f"/data/tmp/{stem}.wav",
    )


async def ignore(_) -> None:
    return None


def test_the_seven_golden_results_are_available():
    assert len(FIXTURE_NAMES) == 7


async def test_demo_stems_select_exactly_that_fixture():
    engine = MockAnalysisEngine(FIXTURES_DIR)
    for name in FIXTURE_NAMES:
        expected = AnalysisResult.model_validate_json(
            (FIXTURES_DIR / f"{name}.result.json").read_text(encoding="utf-8")
        )
        result = await engine.analyze(request(f"demo-{name}"), ignore)
        assert result.chord_events == expected.chord_events
        assert result.metadata == expected.metadata


async def test_other_files_are_spread_over_all_fixtures_by_content_hash():
    engine = MockAnalysisEngine(FIXTURES_DIR)
    seen = set()
    for i in range(len(FIXTURE_NAMES)):
        result = await engine.analyze(request("song", f"{i:064x}"), ignore)
        seen.add(tuple(e.symbol for e in result.chord_events))
    assert len(seen) == len(FIXTURE_NAMES)


async def test_an_unknown_demo_name_falls_back_to_the_hash_choice():
    engine = MockAnalysisEngine(FIXTURES_DIR)
    result = await engine.analyze(request("demo-not-a-fixture"), ignore)
    assert result.chord_events


async def test_fail_ai_is_retryable():
    with pytest.raises(AnalysisError) as exc:
        await MockAnalysisEngine(FIXTURES_DIR).analyze(request("fail-ai"), ignore)
    assert exc.value.retryable is True


async def test_fail_timeout_never_finishes():
    engine = MockAnalysisEngine(FIXTURES_DIR)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(engine.analyze(request("fail-timeout"), ignore), 0.1)


async def test_reports_separating_beat_tracking_and_recognizing():
    seen = []

    async def record(progress):
        seen.append((progress.stage, progress.progress))

    await MockAnalysisEngine(FIXTURES_DIR).analyze(request("song"), record)
    assert seen == [
        (JobStage.SEPARATING, 0.45),
        (JobStage.BEAT_TRACKING, 0.7),
        (JobStage.RECOGNIZING, 0.9),
    ]


async def test_stage_delay_slows_each_step():
    engine = MockAnalysisEngine(FIXTURES_DIR, stage_delay_ms=30)
    started = time.perf_counter()
    await engine.analyze(request("song"), ignore)
    assert time.perf_counter() - started >= 0.09 - 0.01


def test_fixture_files_are_valid_json_documents():
    for path in FIXTURES_DIR.glob("*.json"):
        assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == "1.0"


async def test_files_named_slow_take_longer_per_step_than_the_rest():
    engine = MockAnalysisEngine(FIXTURES_DIR, slow_step_ms=40)

    started = time.perf_counter()
    await engine.analyze(request("slow-song"), ignore)
    slow = time.perf_counter() - started

    started = time.perf_counter()
    await engine.analyze(request("song"), ignore)
    normal = time.perf_counter() - started

    assert slow >= 3 * 0.04 - 0.02
    assert normal < 0.08
