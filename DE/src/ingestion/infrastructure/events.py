"""Job progress fan-out (in-process). A Redis-backed adapter follows with Celery (task 11)."""

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator
from uuid import UUID

from src.analysis_contract.models import JobProgress, JobState

_FINAL = (JobState.DONE, JobState.FAILED)


class InMemoryJobEvents:
    def __init__(self) -> None:
        self._queues: dict[UUID, set[asyncio.Queue[JobProgress]]] = defaultdict(set)

    async def publish(self, progress: JobProgress) -> None:
        for queue in list(self._queues.get(progress.job_id, ())):
            queue.put_nowait(progress)

    def subscribe(self, job_id: UUID) -> AsyncIterator[JobProgress]:
        """Registers immediately (so nothing published after this call is missed) and yields
        updates until the job reaches a final state."""
        queue: asyncio.Queue[JobProgress] = asyncio.Queue()
        self._queues[job_id].add(queue)

        async def stream() -> AsyncIterator[JobProgress]:
            try:
                while True:
                    progress = await queue.get()
                    yield progress
                    if progress.state in _FINAL:
                        return
            finally:
                self._queues[job_id].discard(queue)
                if not self._queues[job_id]:
                    del self._queues[job_id]

        return stream()
