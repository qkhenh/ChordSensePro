from datetime import UTC, datetime
from uuid import uuid4

from src.analysis_contract.models import JobState, SourceKind
from src.ingestion.domain.job import advance, new_job
from src.ingestion.infrastructure.memory import InMemoryJobRepository
from src.ingestion.settings import IngestSettings

NOW = datetime(2026, 10, 26, tzinfo=UTC)


def make_job(title="t"):
    return new_job(owner_id=uuid4(), source_kind=SourceKind.FILE, title=title, now=NOW)


def test_defaults_follow_the_spec_limits():
    s = IngestSettings()
    assert s.max_upload_bytes == 100 * 1024 * 1024
    assert s.max_duration_ms == 600_000
    assert s.job_timeout_s == 90
    assert s.job_runner == "inprocess"


async def test_repository_never_overwrites_a_finished_job():
    repo = InMemoryJobRepository()
    job = await repo.add(make_job())
    await repo.save(advance(job, state=JobState.DONE, chart_id=uuid4(), now=NOW))
    await repo.save(job)  # a late writer still holding the queued snapshot
    assert (await repo.get(job.id)).state == JobState.DONE


async def test_refund_flag_flips_exactly_once():
    repo = InMemoryJobRepository()
    job = await repo.add(make_job())
    assert await repo.try_mark_refunded(job.id) is True
    assert await repo.try_mark_refunded(job.id) is False


async def test_list_unfinished_only_returns_queued_and_processing():
    repo = InMemoryJobRepository()
    open_job = await repo.add(make_job("a"))
    finished = await repo.add(make_job("b"))
    await repo.save(advance(finished, state=JobState.DONE, chart_id=uuid4(), now=NOW))
    assert [j.id for j in await repo.list_unfinished()] == [open_job.id]
