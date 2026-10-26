"""Job progress fan-out (in-process). A Redis-backed adapter follows with Celery (task 11)."""

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

from src.analysis_contract.models import JobProgress, JobState

_FINAL = (JobState.DONE, JobState.FAILED)


class InMemoryJobEvents:
    def __init__(self) -> None:
        self._queues: dict[UUID, set[asyncio.Queue[JobProgress]]] = defaultdict(set)

    def subscriber_count(self, job_id: UUID) -> int:
        return len(self._queues.get(job_id, ()))

    async def publish(self, progress: JobProgress) -> None:
        for queue in list(self._queues.get(progress.job_id, ())):
            queue.put_nowait(progress)

    @asynccontextmanager
    async def subscribe(self, job_id: UUID) -> AsyncIterator[AsyncIterator[JobProgress]]:
        """Registers on entry (nothing published afterwards is missed) and always
        unregisters on exit. The yielded stream ends after a final state."""
        queue: asyncio.Queue[JobProgress] = asyncio.Queue()
        self._queues[job_id].add(queue)

        async def stream() -> AsyncIterator[JobProgress]:
            while True:
                progress = await queue.get()
                yield progress
                if progress.state in _FINAL:
                    return

        try:
            yield stream()
        finally:
            self._queues[job_id].discard(queue)
            if not self._queues[job_id]:
                del self._queues[job_id]
