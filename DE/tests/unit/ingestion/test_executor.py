import asyncio
from dataclasses import replace
from uuid import uuid4

import pytest
import structlog

from src.analysis_contract.models import (
    ErrorCode,
    JobStage,
    JobState,
    SourceKind,
)
from src.identity.domain.models import Role
from src.ingestion.domain.errors import IngestionError
from src.ingestion.infrastructure.mock_engine import MockAnalysisEngine
from tests.support.ingestion import make_world

UNIT_URL = "https://cdn.example.com/a/My%20Song.mp3"


def steps(world):
    return [(p.state, p.stage, p.progress) for p in world.events.published]


async def final(world, job):
    return await world.jobs.get(job.id)


class TestFileHappyPath:
    async def test_runs_every_stage_and_finishes_with_a_chart(self, tmp_path):
        world = make_world(tmp_path)
        job = await world.submit()
        await world.executor.run(job.id)

        assert steps(world) == [
            (JobState.PROCESSING, JobStage.NORMALIZING, 0.15),
            (JobState.PROCESSING, JobStage.SEPARATING, 0.45),
            (JobState.PROCESSING, JobStage.BEAT_TRACKING, 0.7),
            (JobState.PROCESSING, JobStage.RECOGNIZING, 0.9),
            (JobState.DONE, JobStage.RECOGNIZING, 1.0),
        ]
        done = await final(world, job)
        assert done.state == JobState.DONE and done.content_hash
        chart = await world.charts.get(done.result_chart_id, world.user.id)
        assert chart.title == "song"
        assert chart.ai_events and chart.overrides == () and chart.version == 1
        assert chart.source.kind == SourceKind.FILE
        assert chart.source.content_hash == done.content_hash
        assert chart.source.playback.kind == "file"
        assert chart.source.playback.audio_url == f"/api/v1/charts/{chart.id}/audio"
        assert await world.used() == 1  # a successful analysis stays charged

    async def test_keeps_the_canonical_wav_for_playback_and_removes_everything_else(self, tmp_path):
        world = make_world(tmp_path)
        job = await world.submit()
        await world.executor.run(job.id)
        done = await final(world, job)
        assert (tmp_path / "audio" / f"{done.content_hash}.wav").read_bytes() == b"audio-bytes"
        assert not (tmp_path / "tmp" / str(job.id)).exists()

    async def test_the_cache_and_fingerprint_are_filled_for_next_time(self, tmp_path):
        world = make_world(tmp_path)
        job = await world.submit(fingerprint="f" * 64)
        await world.executor.run(job.id)
        done = await final(world, job)
        assert (await world.cache.get(done.content_hash)).content_hash == done.content_hash
        assert await world.cache.fingerprint_to_hash("f" * 64) == done.content_hash

    async def test_every_change_is_published_after_it_is_saved(self, tmp_path):
        world = make_world(tmp_path)
        job = await world.submit()
        await world.executor.run(job.id)
        assert world.events.published[-1].state == JobState.DONE
        assert (
            world.events.published[-1].result_chart_id == (await final(world, job)).result_chart_id
        )


class TestUrlHappyPath:
    async def test_fetches_first_and_uses_the_link_for_playback(self, tmp_path):
        world = make_world(tmp_path)
        job = await world.submit(kind=SourceKind.URL, url=UNIT_URL, name="My Song")
        await world.executor.run(job.id)

        assert steps(world)[:2] == [
            (JobState.PROCESSING, JobStage.FETCHING, 0.05),
            (JobState.PROCESSING, JobStage.NORMALIZING, 0.15),
        ]
        done = await final(world, job)
        chart = await world.charts.get(done.result_chart_id, world.user.id)
        assert chart.title == "My Song"
        assert chart.source.url == UNIT_URL
        assert chart.source.playback.kind == "direct"
        assert chart.source.playback.audio_url == UNIT_URL
        # Q2: the server keeps no audio for a link, only the player embeds/plays from the origin
        assert not (tmp_path / "audio").exists() or not list((tmp_path / "audio").iterdir())
        assert not (tmp_path / "tmp" / str(job.id)).exists()


class TestFailures:
    async def test_an_analysis_error_fails_the_job_and_refunds_once(self, tmp_path):
        world = make_world(tmp_path)
        job = await world.submit(name="fail-ai.mp3")
        assert await world.used() == 1

        await world.executor.run(job.id)
        failed = await final(world, job)
        assert failed.state == JobState.FAILED
        assert (failed.error.code, failed.error.retryable) == (ErrorCode.AI_FAILURE, True)
        assert failed.result_chart_id is None
        assert await world.used() == 0

        await world.executor.run(job.id)  # a duplicate delivery changes nothing
        assert await world.used() == 0
        assert world.events.published[-1].state == JobState.FAILED
        assert [p.state for p in world.events.published].count(JobState.FAILED) == 1

    async def test_a_timeout_fails_the_job_as_analysis_timeout(self, tmp_path):
        world = make_world(tmp_path, job_timeout_s=0.05)
        job = await world.submit(name="fail-timeout.mp3")
        await world.executor.run(job.id)
        failed = await final(world, job)
        assert failed.error.code == ErrorCode.ANALYSIS_TIMEOUT
        assert await world.used() == 0

    async def test_an_unexpected_exception_is_contained_logged_and_not_leaked(self, tmp_path):
        class Crashing:
            async def analyze(self, request, on_progress):
                raise RuntimeError("secret internal detail")

        world = make_world(tmp_path, engine=Crashing())
        job = await world.submit()
        with structlog.testing.capture_logs() as logs:
            await world.executor.run(job.id)
        failed = await final(world, job)
        assert failed.state == JobState.FAILED
        assert failed.error.code == ErrorCode.AI_FAILURE
        assert "secret" not in failed.error.message
        assert all("secret" not in p.model_dump_json() for p in world.events.published)
        assert any(e.get("job_id") == str(job.id) and e["log_level"] == "error" for e in logs)
        assert await world.used() == 0
        assert not (tmp_path / "tmp" / str(job.id)).exists()

    async def test_resolver_and_normalizer_errors_keep_their_codes(self, tmp_path):
        world = make_world(tmp_path)
        job = await world.submit(
            kind=SourceKind.URL, url="https://example.com/fail-unresolvable.mp3"
        )
        await world.executor.run(job.id)
        assert (await final(world, job)).error.code == ErrorCode.URL_UNRESOLVABLE

        async def refuse(src, dest_dir, stem):
            raise IngestionError(ErrorCode.UNSUPPORTED_FORMAT, "not audio")

        world.normalizer.normalize = refuse
        second = await world.submit()
        await world.executor.run(second.id)
        assert (await final(world, second)).error.code == ErrorCode.UNSUPPORTED_FORMAT
        assert await world.used() == 0

    async def test_an_unlimited_tier_is_never_refunded(self, tmp_path):
        world = make_world(tmp_path, role=Role.PAID)
        job = await world.submit(name="fail-ai.mp3")
        assert job.quota_charged is False
        await world.executor.run(job.id)
        assert (await final(world, job)).state == JobState.FAILED
        assert await world.used() == 0


class TestCacheHit:
    async def test_same_waveform_skips_the_engine_and_hands_the_quota_back(self, tmp_path):
        first = make_world(tmp_path)
        seed = await first.submit()
        await first.executor.run(seed.id)
        cached = await first.cache.get((await final(first, seed)).content_hash)

        class MustNotRun:
            async def analyze(self, request, on_progress):
                raise AssertionError("the engine must not run on a cache hit")

        world = make_world(tmp_path / "other", engine=MustNotRun())
        await world.cache.put(cached)
        job = await world.submit()
        assert await world.used() == 1
        await world.executor.run(job.id)

        done = await final(world, job)
        assert done.state == JobState.DONE
        assert await world.used() == 0  # charged at acceptance, returned: same content
        chart = await world.charts.get(done.result_chart_id, world.user.id)
        assert chart.ai_events == cached.chord_events
        assert chart.source.content_hash == cached.content_hash


class TestRecovery:
    async def test_stale_unfinished_jobs_are_failed_and_refunded(self, tmp_path):
        world = make_world(tmp_path, job_timeout_s=90)
        stale = await world.submit()
        world.clock.advance(seconds=200)
        fresh = await world.submit()

        assert await world.executor.recover_orphans() == 1
        assert (await final(world, stale)).state == JobState.FAILED
        assert (await final(world, stale)).error.code == ErrorCode.AI_FAILURE
        assert (await final(world, fresh)).state == JobState.QUEUED
        assert await world.used() == 1  # only the fresh job still holds a charge
        assert world.events.published[-1].state == JobState.FAILED

    async def test_a_processing_job_is_recovered_too_and_its_work_dir_removed(self, tmp_path):
        world = make_world(tmp_path)
        job = await world.submit()
        await world.jobs.save(
            replace(job, state=JobState.PROCESSING, stage=JobStage.SEPARATING, progress=0.45)
        )
        world.clock.advance(seconds=95)
        assert await world.executor.recover_orphans() == 1
        assert (await final(world, job)).state == JobState.FAILED
        assert not (tmp_path / "tmp" / str(job.id)).exists()


class TestEdges:
    async def test_running_an_unknown_or_finished_job_is_a_no_op(self, tmp_path):
        world = make_world(tmp_path)
        await world.executor.run(uuid4())
        job = await world.submit()
        await world.executor.run(job.id)
        count = len(world.events.published)
        await world.executor.run(job.id)
        assert len(world.events.published) == count

    async def test_cancellation_propagates_instead_of_being_swallowed(self, tmp_path):
        started = asyncio.Event()

        class Hanging(MockAnalysisEngine):
            async def analyze(self, request, on_progress):
                started.set()
                await asyncio.sleep(3600)

        world = make_world(tmp_path, engine=Hanging())
        job = await world.submit()
        task = asyncio.create_task(world.executor.run(job.id))
        await asyncio.wait_for(started.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
