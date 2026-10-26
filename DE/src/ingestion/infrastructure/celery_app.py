"""Celery adapter of the job runner.

The API process only publishes ``ingestion.run_job(job_id)``; a worker builds its own services
for every task (``configure``) and runs the same ``JobExecutor`` the in-process runner uses.
Run a worker with ``celery -A src.ingestion.worker worker --pool=solo`` (prefork does not exist
on Windows).
"""

import asyncio
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from uuid import UUID

from celery import Celery

from src.ingestion.application.executor import JobExecutor

TASK_NAME = "ingestion.run_job"
# Hard limits sit above the executor's own asyncio timeout, as a backstop.
SOFT_LIMIT_MARGIN_S = 5
HARD_LIMIT_MARGIN_S = 15

ExecutorFactory = Callable[[], AbstractAsyncContextManager[JobExecutor]]
_factory: ExecutorFactory | None = None


def configure(factory: ExecutorFactory | None) -> None:
    """Tell tasks how to obtain an executor (an async context manager that cleans up after).

    A new executor per task, because each task runs in its own event loop and database
    connections must not outlive it."""
    global _factory
    _factory = factory


async def _run(job_id: str) -> None:
    if _factory is None:
        raise RuntimeError("no executor factory configured: call celery_app.configure() first")
    async with _factory() as executor:
        await executor.run(UUID(job_id))


def make_celery(broker_url: str, *, job_timeout_s: float, eager: bool = False) -> Celery:
    app = Celery("chordsense", broker=broker_url)
    app.conf.update(
        task_ignore_result=True,
        task_always_eager=eager,
        task_eager_propagates=False,
        broker_connection_retry_on_startup=True,
        broker_transport_options={"max_retries": 2, "socket_connect_timeout": 3},
        task_publish_retry_policy={"max_retries": 2, "interval_start": 0.1, "interval_step": 0.2},
    )

    @app.task(
        name=TASK_NAME,
        soft_time_limit=job_timeout_s + SOFT_LIMIT_MARGIN_S,
        time_limit=job_timeout_s + HARD_LIMIT_MARGIN_S,
    )
    def run_job(job_id: str) -> None:
        asyncio.run(_run(job_id))

    return app


class CeleryJobRunner:
    def __init__(self, app: Celery) -> None:
        self._task = app.tasks[TASK_NAME]

    async def enqueue(self, job_id: UUID) -> None:
        # Publishing blocks on the network (and, in eager mode, runs the job): keep it off the loop.
        await asyncio.to_thread(self._task.delay, str(job_id))

    async def drain(self) -> None:
        """Nothing to wait for: the work happens in other processes."""
        return None
