"""Ingestion repositories against real PostgreSQL (database *_test)."""

import asyncio
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from src.analysis_contract.models import (
    AnalysisResult,
    ErrorCode,
    JobError,
    JobStage,
    JobState,
    SourceKind,
)
from src.identity.infrastructure.postgres import PgUserRepository
from src.ingestion.application.charts import build_chart
from src.ingestion.domain.job import advance, new_job
from src.ingestion.infrastructure.mock_engine import FIXTURES_DIR
from src.ingestion.infrastructure.postgres import (
    PgAnalysisCache,
    PgChartRepository,
    PgJobRepository,
)
from tests.support.ingestion import make_user

pytestmark = pytest.mark.integration

DE_DIR = Path(__file__).parents[2]
NOW = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)
RESULT = AnalysisResult.model_validate_json(
    (FIXTURES_DIR / "ii_v_i_c.result.json").read_text(encoding="utf-8")
)


@pytest.fixture(scope="module", autouse=True)
def migrated():
    url = os.environ["TEST_DATABASE_URL"]
    assert url.rsplit("/", 1)[-1].endswith("_test"), "refusing to touch a non-test database"
    cfg = Config(str(DE_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(DE_DIR / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url)
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")


@pytest.fixture
async def engine() -> AsyncEngine:
    engine = create_async_engine(os.environ["TEST_DATABASE_URL"])
    async with engine.begin() as conn:
        await conn.exec_driver_sql("TRUNCATE app.users, app.analyses CASCADE")
    yield engine
    await engine.dispose()


@pytest.fixture
async def owner(engine):
    return await PgUserRepository(engine).add(make_user())


def queued_job(owner, **kwargs):
    return new_job(owner_id=owner.id, source_kind=SourceKind.FILE, title="Song", now=NOW, **kwargs)


class TestPgJobRepository:
    async def test_round_trip_keeps_every_field(self, engine, owner):
        repo = PgJobRepository(engine)
        job = queued_job(
            owner, input_path="/data/tmp/x/song.mp3", fingerprint="a" * 64, charged_on=NOW.date()
        )
        await repo.add(job)
        assert await repo.get(job.id) == job

    async def test_a_cache_hit_job_can_be_added_already_done(self, engine, owner):
        repo = PgJobRepository(engine)
        chart = build_chart(queued_job(owner), RESULT, now=NOW)
        await PgChartRepository(engine).add(chart)
        done = advance(queued_job(owner), state=JobState.DONE, chart_id=chart.id, now=NOW)
        await repo.add(done)
        stored = await repo.get(done.id)
        assert (stored.state, stored.progress, stored.result_chart_id) == (
            JobState.DONE,
            1.0,
            chart.id,
        )
        assert await repo.list_unfinished() == []

    async def test_unknown_job_is_none(self, engine):
        assert await PgJobRepository(engine).get(uuid4()) is None

    async def test_progress_and_stage_survive_a_save(self, engine, owner):
        repo = PgJobRepository(engine)
        job = await repo.add(queued_job(owner))
        moved = advance(
            job, state=JobState.PROCESSING, stage=JobStage.SEPARATING, progress=0.45, now=NOW
        )
        await repo.save(moved)
        stored = await repo.get(job.id)
        assert (stored.state, stored.stage, stored.progress) == (
            JobState.PROCESSING,
            JobStage.SEPARATING,
            0.45,
        )

    async def test_a_failed_job_keeps_its_error(self, engine, owner):
        repo = PgJobRepository(engine)
        job = await repo.add(queued_job(owner))
        error = JobError(code=ErrorCode.ANALYSIS_TIMEOUT, message="slow", retryable=True)
        await repo.save(advance(job, state=JobState.FAILED, error=error, now=NOW))
        stored = await repo.get(job.id)
        assert (stored.state, stored.error) == (JobState.FAILED, error)

    async def test_a_finished_job_is_never_overwritten(self, engine, owner):
        repo = PgJobRepository(engine)
        job = await repo.add(queued_job(owner))
        error = JobError(code=ErrorCode.AI_FAILURE, message="x")
        await repo.save(advance(job, state=JobState.FAILED, error=error, now=NOW))
        await repo.save(job)  # a late writer holding the queued snapshot
        assert (await repo.get(job.id)).state == JobState.FAILED

    async def test_list_unfinished_returns_queued_and_processing_only(self, engine, owner):
        repo = PgJobRepository(engine)
        waiting = await repo.add(queued_job(owner))
        running = await repo.add(queued_job(owner))
        await repo.save(
            advance(running, state=JobState.PROCESSING, stage=JobStage.NORMALIZING, now=NOW)
        )
        failed = await repo.add(queued_job(owner))
        await repo.save(
            advance(
                failed,
                state=JobState.FAILED,
                error=JobError(code=ErrorCode.AI_FAILURE, message="x"),
                now=NOW,
            )
        )
        assert {j.id for j in await repo.list_unfinished()} == {waiting.id, running.id}

    async def test_refund_flag_flips_exactly_once_even_under_concurrency(self, engine, owner):
        repo = PgJobRepository(engine)
        job = await repo.add(queued_job(owner))
        outcomes = await asyncio.gather(*(repo.try_mark_refunded(job.id) for _ in range(10)))
        assert outcomes.count(True) == 1
        assert (await repo.get(job.id)).quota_charged is False

    async def test_saving_never_resurrects_a_flipped_refund_flag(self, engine, owner):
        repo = PgJobRepository(engine)
        job = await repo.add(queued_job(owner))
        await repo.try_mark_refunded(job.id)
        await repo.save(
            advance(job, state=JobState.PROCESSING, stage=JobStage.NORMALIZING, now=NOW)
        )
        assert (await repo.get(job.id)).quota_charged is False


class TestPgChartRepository:
    async def test_round_trip_and_owner_isolation(self, engine, owner):
        repo = PgChartRepository(engine)
        chart = build_chart(queued_job(owner), RESULT, now=NOW)
        await repo.add(chart)
        assert await repo.get(chart.id, owner.id) == chart
        assert await repo.get(chart.id, uuid4()) is None
        assert await repo.get(uuid4(), owner.id) is None

    async def test_deleting_the_owner_removes_the_chart(self, engine, owner):
        repo = PgChartRepository(engine)
        chart = build_chart(queued_job(owner), RESULT, now=NOW)
        await repo.add(chart)
        async with engine.begin() as conn:
            await conn.exec_driver_sql("TRUNCATE app.users CASCADE")
        assert await repo.get(chart.id, owner.id) is None


class TestPgAnalysisCache:
    async def test_put_get_round_trip(self, engine):
        cache = PgAnalysisCache(engine)
        assert await cache.get(RESULT.content_hash) is None
        await cache.put(RESULT)
        assert await cache.get(RESULT.content_hash) == RESULT

    async def test_put_twice_keeps_the_first_result(self, engine):
        cache = PgAnalysisCache(engine)
        await cache.put(RESULT)
        await cache.put(RESULT.model_copy(update={"job_id": uuid4()}))
        assert (await cache.get(RESULT.content_hash)).job_id == RESULT.job_id

    async def test_fingerprints_point_at_a_cached_result(self, engine):
        cache = PgAnalysisCache(engine)
        await cache.put(RESULT)
        assert await cache.fingerprint_to_hash("c" * 64) is None
        await cache.remember_fingerprint("c" * 64, RESULT.content_hash)
        await cache.remember_fingerprint("c" * 64, RESULT.content_hash)  # idempotent
        assert await cache.fingerprint_to_hash("c" * 64) == RESULT.content_hash
