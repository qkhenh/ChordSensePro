"""Celery job runner in eager mode (no broker) and the wiring that selects it."""

from contextlib import asynccontextmanager
from uuid import UUID, uuid4

import pytest
from celery.exceptions import SoftTimeLimitExceeded
from sqlalchemy.ext.asyncio import create_async_engine

from src.analysis_contract.models import ErrorCode, JobState
from src.api.wiring import build_ingestion
from src.ingestion.infrastructure.celery_app import CeleryJobRunner, configure, make_celery
from src.ingestion.infrastructure.events import InMemoryJobEvents
from src.ingestion.infrastructure.events_redis import RedisJobEvents
from src.ingestion.infrastructure.runner import InProcessJobRunner
from src.ingestion.settings import IngestSettings
from tests.support.identity import make_container
from tests.support.ingestion import make_world


class RecordingExecutor:
    def __init__(self):
        self.ran: list = []

    async def run(self, job_id):
        self.ran.append(job_id)


def use_executor(executor):
    @asynccontextmanager
    async def factory():
        yield executor

    configure(factory)


def eager_app(job_timeout_s: float = 90):
    return make_celery("memory://", job_timeout_s=job_timeout_s, eager=True)


async def test_enqueue_runs_the_executor_once_with_the_job_id():
    executor = RecordingExecutor()
    use_executor(executor)
    runner = CeleryJobRunner(eager_app())
    job_id = uuid4()
    await runner.enqueue(job_id)
    assert executor.ran == [job_id]
    assert isinstance(executor.ran[0], UUID)  # the id crosses the broker as a string


async def test_every_enqueue_is_its_own_run():
    executor = RecordingExecutor()
    use_executor(executor)
    runner = CeleryJobRunner(eager_app())
    ids = [uuid4(), uuid4(), uuid4()]
    for job_id in ids:
        await runner.enqueue(job_id)
    assert executor.ran == ids


async def test_a_celery_time_limit_fails_the_job_as_a_timeout_and_refunds(tmp_path):
    class Hung:
        async def analyze(self, request, on_progress):
            raise SoftTimeLimitExceeded  # what Celery raises inside a task that ran too long

    world = make_world(
        tmp_path, engine=Hung(), timeout_errors=(TimeoutError, SoftTimeLimitExceeded)
    )
    use_executor(world.executor)
    job = await world.submit()
    assert await world.used() == 1

    await CeleryJobRunner(eager_app()).enqueue(job.id)

    failed = await world.jobs.get(job.id)
    assert failed.state == JobState.FAILED
    assert failed.error.code == ErrorCode.ANALYSIS_TIMEOUT
    assert await world.used() == 0


def test_celery_limits_follow_the_analysis_timeout():
    task = eager_app(job_timeout_s=90).tasks["ingestion.run_job"]
    assert (task.soft_time_limit, task.time_limit) == (95, 105)


async def test_drain_has_nothing_to_wait_for():
    assert await CeleryJobRunner(eager_app()).drain() is None


class TestWiring:
    def container(self):
        return make_container()

    def engine(self):
        return create_async_engine("postgresql+asyncpg://nobody@127.0.0.1:1/none")

    def test_the_default_runner_lives_in_the_api_process(self, tmp_path):
        ingestion = build_ingestion(
            IngestSettings(data_dir=tmp_path), self.engine(), self.container()
        )
        assert isinstance(ingestion.runner, InProcessJobRunner)
        assert isinstance(ingestion.events, InMemoryJobEvents)

    def test_celery_selects_the_broker_runner_and_redis_events(self, tmp_path):
        settings = IngestSettings(data_dir=tmp_path, job_runner="celery")
        ingestion = build_ingestion(settings, self.engine(), self.container())
        assert isinstance(ingestion.runner, CeleryJobRunner)
        assert isinstance(ingestion.events, RedisJobEvents)
        assert SoftTimeLimitExceeded in ingestion.executor._timeout_errors


@pytest.fixture(autouse=True)
def _reset_factory():
    yield
    configure(None)
