"""Job progress over Redis pub/sub: lets the API process see what a Celery worker reports."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

import redis.asyncio as aioredis
import structlog
from pydantic import ValidationError

from src.analysis_contract.models import JobProgress, JobState

log = structlog.get_logger("ingestion")
_FINAL = (JobState.DONE, JobState.FAILED)
_ACK_TIMEOUT_S = 5


def _channel(job_id: UUID) -> str:
    return f"jobs:{job_id}"


class RedisJobEvents:
    def __init__(self, url: str) -> None:
        # protocol=2: the Redis builds common on Windows dev machines predate RESP3 (HELLO).
        self._redis = aioredis.from_url(url, protocol=2)

    async def aclose(self) -> None:
        await self._redis.aclose()

    async def publish(self, progress: JobProgress) -> None:
        await self._redis.publish(_channel(progress.job_id), progress.model_dump_json())

    @asynccontextmanager
    async def subscribe(self, job_id: UUID) -> AsyncIterator[AsyncIterator[JobProgress]]:
        pubsub = self._redis.pubsub()
        channel = _channel(job_id)
        await pubsub.subscribe(channel)
        # Wait for the server's acknowledgement: only then is nothing published after this
        # point missed.
        while True:
            ack = await pubsub.get_message(ignore_subscribe_messages=False, timeout=_ACK_TIMEOUT_S)
            if ack is None:
                await pubsub.aclose()
                raise TimeoutError("Redis did not confirm the subscription")
            if ack["type"] == "subscribe":
                break

        async def stream() -> AsyncIterator[JobProgress]:
            async for message in pubsub.listen():
                if message["type"] != "message":
                    continue
                try:
                    progress = JobProgress.model_validate_json(message["data"])
                except ValidationError:
                    log.warning("ignored_malformed_job_event", channel=channel)
                    continue
                yield progress
                if progress.state in _FINAL:
                    return

        try:
            yield stream()
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()
