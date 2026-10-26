"""The app sweeps stale temporary directories at startup and keeps doing so in the background."""

import os
from dataclasses import replace
from uuid import uuid4

from src.analysis_contract.models import JobStage, JobState, SourceKind
from src.api.app import create_app
from src.api.settings import ApiSettings
from src.ingestion.domain.job import new_job
from src.ingestion.infrastructure.ffmpeg import FfmpegNormalizer
from tests.support.identity import FakeClock, make_container
from tests.support.ingestion import build_ingestion, make_user


async def test_startup_removes_stale_work_directories_and_shutdown_stops_the_sweeper(tmp_path):
    clock = FakeClock()
    container = make_container(clock)
    ingestion = build_ingestion(
        tmp_path / "data", quota=container.quota, clock=clock, normalizer=FfmpegNormalizer()
    )
    stale = tmp_path / "data" / "tmp" / str(uuid4())
    stale.mkdir(parents=True)
    (stale / "upload.bin").write_bytes(b"x")
    long_ago = clock().timestamp() - 10 * ingestion.settings.temp_ttl_s
    os.utime(stale, (long_ago, long_ago))

    app = create_app(container, ApiSettings(), ingestion=ingestion)
    async with app.router.lifespan_context(app):
        assert not stale.exists()
        assert not app.state.sweeper.done()
    assert app.state.sweeper.done()


async def test_an_app_without_ingestion_services_still_starts_and_stops():
    app = create_app(make_container(FakeClock()), ApiSettings())
    async with app.router.lifespan_context(app):
        assert not hasattr(app.state, "sweeper")


async def test_startup_fails_jobs_a_crashed_process_left_behind_even_fresh_ones(tmp_path):
    clock = FakeClock()
    container = make_container(clock)
    ingestion = build_ingestion(
        tmp_path / "data", quota=container.quota, clock=clock, normalizer=FfmpegNormalizer()
    )
    user = make_user()
    receipt = await container.quota.consume(user)
    job = await ingestion.jobs.add(
        new_job(
            owner_id=user.id,
            source_kind=SourceKind.FILE,
            title="t",
            now=clock(),
            quota_charged=receipt.charged,
            charged_on=receipt.day,
        )
    )
    await ingestion.jobs.save(
        replace(job, state=JobState.PROCESSING, stage=JobStage.SEPARATING, progress=0.45)
    )
    clock.advance(seconds=5)  # restarted 5 s after the crash: far below the 300 s deadline
    assert (await container.quota.status(user)).used == 1

    app = create_app(container, ApiSettings(), ingestion=ingestion)
    async with app.router.lifespan_context(app):
        assert (await ingestion.jobs.get(job.id)).state == JobState.FAILED
        assert (await container.quota.status(user)).used == 0
        assert not app.state.recovery.done()
    assert app.state.recovery.done()
