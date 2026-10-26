"""Every shared fixture in contracts/fixtures must parse with the Pydantic contract.

The same files are consumed by the frontend (MSW handlers, Vitest) and by the mock
AnalysisEngine, so a fixture that stops validating breaks all three at once.
"""

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from src.analysis_contract import (
    AnalysisRequest,
    AnalysisResult,
    Chart,
    ErrorCode,
    JobProgress,
    JobState,
    apply_overrides,
)

FIXTURES = Path(__file__).parents[4] / "contracts" / "fixtures"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def files(folder: str, pattern: str = "*.json"):
    found = sorted((FIXTURES / folder).glob(pattern))
    assert found, f"no fixtures in contracts/fixtures/{folder}"
    return found


def ids(paths):
    return [p.name for p in paths]


RESULTS = files("results")
CHARTS = files("charts", "*.chart.json")
REQUESTS = files("requests")


@pytest.mark.parametrize("path", RESULTS, ids=ids(RESULTS))
def test_result_fixture_is_valid(path):
    assert AnalysisResult.model_validate(load(path)).chord_events


@pytest.mark.parametrize("path", CHARTS, ids=ids(CHARTS))
def test_chart_fixture_is_valid_and_effective_timeline_is_consistent(path):
    chart = Chart.model_validate(load(path))
    effective = apply_overrides(chart.ai_events, chart.overrides)
    assert [e.start_ms for e in effective] == sorted(e.start_ms for e in effective)


@pytest.mark.parametrize("path", REQUESTS, ids=ids(REQUESTS))
def test_request_fixture_is_valid(path):
    AnalysisRequest.model_validate(load(path))


def test_success_job_sequence_walks_the_lifecycle():
    seq = TypeAdapter(list[JobProgress]).validate_python(
        load(FIXTURES / "jobs" / "success.sequence.json")
    )
    assert seq[0].state is JobState.QUEUED
    assert seq[-1].state is JobState.DONE
    assert [p.progress for p in seq] == sorted(p.progress for p in seq)
    assert len({p.job_id for p in seq}) == 1


def test_there_is_a_failed_job_fixture_for_every_error_code():
    raw = load(FIXTURES / "jobs" / "errors.json")
    parsed = {code: JobProgress.model_validate(v) for code, v in raw.items()}
    assert set(parsed) == {c.value for c in ErrorCode}
    for code, progress in parsed.items():
        assert progress.state is JobState.FAILED
        assert progress.error is not None
        assert progress.error.code.value == code


def test_fixture_set_covers_the_cases_the_frontend_grid_depends_on():
    results = {
        p.stem.removesuffix(".result"): AnalysisResult.model_validate(load(p)) for p in RESULTS
    }
    grids = [r.metadata.beat_grid for r in results.values()]
    assert any(g.first_beat_offset_ms > 0 for g in grids), "needs a non-zero first-beat offset"
    assert any(g.time_signature.numerator == 3 for g in grids), "needs 3/4"
    assert any(g.time_signature.denominator == 8 for g in grids), "needs a compound meter"
    assert any(g.downbeats_ms for g in grids), "needs explicit downbeats (irregular tempo)"
    assert any(e.chord.bass is not None for r in results.values() for e in r.chord_events), (
        "needs a slash chord"
    )
    symbols = {e.symbol for r in results.values() for e in r.chord_events}
    assert {"Am7b5", "A#maj7", "G/B", "Cmaj9"} <= symbols
