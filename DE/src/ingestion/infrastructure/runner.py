"""In-process job runner: asyncio tasks inside the API process (dev and tests)."""

import asyncio
from typing import Protocol
from uuid import UUID


class _Runs(Protocol):
    async def run(self, job_id: UUID) -> None: ...


class InProcessJobRunner:
    def __init__(self, executor: _Runs) -> None:
        self._executor = executor
        self._tasks: set[asyncio.Task] = set()

    @property
    def pending(self) -> int:
        return len(self._tasks)

    async def enqueue(self, job_id: UUID) -> None:
        # The loop keeps only weak references to tasks: hold them until they finish.
        task = asyncio.create_task(self._executor.run(job_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def drain(self) -> None:
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)
