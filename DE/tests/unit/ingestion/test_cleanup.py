import asyncio
import os
from uuid import UUID, uuid4

import pytest

from src.analysis_contract.models import (
    ErrorCode,
    JobError,
    JobStage,
    JobState,
    SourceKind,
)
from src.ingestion.application.cleanup import TempSweeper
from src.ingestion.domain.job import advance, new_job
from src.ingestion.infrastructure.memory import InMemoryJobRepository
from tests.support.identity import FakeClock

TTL = 3600


class Setup:
    def __init__(self, tmp_path):
        self.data = tmp_path / "data"
        self.root = self.data / "tmp"
        self.root.mkdir(parents=True)
        self.clock = FakeClock()
        self.jobs = InMemoryJobRepository()
        self.sweeper = TempSweeper(self.root, TTL, self.jobs, self.clock)

    def make_dir(self, name: str | UUID, age_s: float, files: int = 2):
        directory = self.root / str(name)
        directory.mkdir()
        for i in range(files):
            (directory / f"f{i}.bin").write_bytes(b"x" * 10)
        stamp = self.clock().timestamp() - age_s
        os.utime(directory, (stamp, stamp))
        return directory

    async def job(self, state: JobState = JobState.QUEUED):
        job = await self.jobs.add(
            new_job(owner_id=uuid4(), source_kind=SourceKind.FILE, title="t", now=self.clock())
        )
        if state == JobState.PROCESSING:
            job = advance(
                job,
                state=JobState.PROCESSING,
                stage=JobStage.NORMALIZING,
                now=self.clock(),
            )
        elif state == JobState.DONE:
            job = advance(job, state=JobState.DONE, chart_id=uuid4(), now=self.clock())
        elif state == JobState.FAILED:
            job = advance(
                job,
                state=JobState.FAILED,
                error=JobError(code=ErrorCode.AI_FAILURE, message="x"),
                now=self.clock(),
            )
        await self.jobs.save(job)
        return job


@pytest.fixture
def env(tmp_path):
    return Setup(tmp_path)


async def test_old_directories_of_finished_jobs_are_removed(env):
    done = await env.job(JobState.DONE)
    failed = await env.job(JobState.FAILED)
    a = env.make_dir(done.id, TTL + 60)
    b = env.make_dir(failed.id, TTL + 60)
    assert await env.sweeper.sweep() == 2
    assert not a.exists() and not b.exists()


@pytest.mark.parametrize("state", [JobState.QUEUED, JobState.PROCESSING])
async def test_directories_of_unfinished_jobs_are_kept_however_old(env, state):
    job = await env.job(state)
    directory = env.make_dir(job.id, 10 * TTL)
    assert await env.sweeper.sweep() == 0
    assert directory.exists()


async def test_recent_directories_are_kept(env):
    job = await env.job(JobState.DONE)
    directory = env.make_dir(job.id, TTL - 60)
    assert await env.sweeper.sweep() == 0
    assert directory.exists()


async def test_leftovers_of_jobs_that_never_existed_are_removed_once_old(env):
    stale = env.make_dir(uuid4(), TTL + 1)
    fresh = env.make_dir(uuid4(), 5)
    assert await env.sweeper.sweep() == 1
    assert not stale.exists() and fresh.exists()


async def test_only_uuid_named_directories_are_ever_touched(env):
    stranger = env.make_dir("not-a-job", 10 * TTL)
    note = env.root / "README.txt"
    note.write_text("hello")
    os.utime(note, (0, 0))
    assert await env.sweeper.sweep() == 0
    assert stranger.exists() and note.exists()


async def test_siblings_of_the_tmp_directory_are_never_swept(env):
    audio = env.data / "audio"
    demo = env.data / "demo"
    for folder in (audio, demo):
        folder.mkdir()
        (folder / "keep.wav").write_bytes(b"RIFF")
        os.utime(folder, (0, 0))
        os.utime(folder / "keep.wav", (0, 0))
    await env.sweeper.sweep()
    assert (audio / "keep.wav").exists() and (demo / "keep.wav").exists()


async def test_a_symlink_is_skipped_and_its_target_survives(env, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "precious.txt").write_text("keep me")
    link = env.root / str(uuid4())
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks need privileges on this machine")
    os.utime(outside, (0, 0))
    assert await env.sweeper.sweep() == 0
    assert (outside / "precious.txt").read_text() == "keep me"


async def test_a_missing_root_is_not_an_error(tmp_path):
    sweeper = TempSweeper(tmp_path / "nope", TTL, InMemoryJobRepository(), FakeClock())
    assert await sweeper.sweep() == 0


async def test_run_forever_keeps_sweeping_and_survives_a_failing_pass(env):
    calls = {"n": 0}
    real = env.jobs.get

    async def flaky(job_id):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("database hiccup")
        return await real(job_id)

    env.jobs.get = flaky
    env.make_dir(uuid4(), TTL + 1)  # unknown job: needs a lookup, which fails once

    task = asyncio.create_task(env.sweeper.run_forever(interval_s=0.01))
    try:
        for _ in range(100):
            if not list(env.root.iterdir()):
                break
            await asyncio.sleep(0.02)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert list(env.root.iterdir()) == []
    assert calls["n"] >= 2
