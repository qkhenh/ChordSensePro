import asyncio
import gc
from uuid import uuid4

from src.ingestion.infrastructure.runner import InProcessJobRunner


class SlowExecutor:
    def __init__(self):
        self.release = asyncio.Event()
        self.ran: list = []

    async def run(self, job_id):
        await self.release.wait()
        self.ran.append(job_id)


class ExplodingExecutor:
    async def run(self, job_id):
        raise RuntimeError("executor bug")


async def test_enqueue_returns_before_the_job_runs():
    executor = SlowExecutor()
    runner = InProcessJobRunner(executor)
    job_id = uuid4()
    await runner.enqueue(job_id)
    assert executor.ran == []
    executor.release.set()
    await runner.drain()
    assert executor.ran == [job_id]


async def test_running_jobs_are_not_garbage_collected():
    executor = SlowExecutor()
    runner = InProcessJobRunner(executor)
    await runner.enqueue(uuid4())
    gc.collect()
    executor.release.set()
    await runner.drain()
    assert len(executor.ran) == 1


async def test_drain_waits_for_every_job_and_finished_tasks_are_forgotten():
    executor = SlowExecutor()
    runner = InProcessJobRunner(executor)
    for _ in range(3):
        await runner.enqueue(uuid4())
    executor.release.set()
    await runner.drain()
    assert len(executor.ran) == 3
    assert runner.pending == 0


async def test_a_crashing_job_does_not_break_the_runner():
    runner = InProcessJobRunner(ExplodingExecutor())
    await runner.enqueue(uuid4())
    await runner.drain()  # must not raise
    assert runner.pending == 0
