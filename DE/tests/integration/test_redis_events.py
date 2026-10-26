"""RedisJobEvents against a real Redis (TEST_REDIS_URL): the bridge worker -> API process."""

import asyncio
import os
from uuid import uuid4

import pytest
import redis.asyncio as aioredis

from src.analysis_contract.models import JobProgress, JobStage, JobState
from src.ingestion.infrastructure.events_redis import RedisJobEvents

pytestmark = pytest.mark.redis


def processing(job_id, progress: float, stage=JobStage.SEPARATING) -> JobProgress:
    return JobProgress(job_id=job_id, state=JobState.PROCESSING, stage=stage, progress=progress)


def done(job_id) -> JobProgress:
    return JobProgress(job_id=job_id, state=JobState.DONE, progress=1.0, result_chart_id=uuid4())


@pytest.fixture
async def events():
    bus = RedisJobEvents(os.environ["TEST_REDIS_URL"])
    yield bus
    await bus.aclose()


async def collect(stream) -> list[JobProgress]:
    return [item async for item in stream]


async def test_a_second_client_sees_the_published_changes_in_order(events):
    other = RedisJobEvents(os.environ["TEST_REDIS_URL"])  # the worker's connection
    job_id = uuid4()
    try:
        async with events.subscribe(job_id) as stream:
            await other.publish(processing(job_id, 0.45))
            await other.publish(processing(job_id, 0.7, JobStage.BEAT_TRACKING))
            await other.publish(done(job_id))
            seen = await asyncio.wait_for(collect(stream), 5)
    finally:
        await other.aclose()
    assert [(p.state, p.progress) for p in seen] == [
        (JobState.PROCESSING, 0.45),
        (JobState.PROCESSING, 0.7),
        (JobState.DONE, 1.0),
    ]


async def test_nothing_published_right_after_entering_is_missed(events):
    job_id = uuid4()
    async with events.subscribe(job_id) as stream:
        await events.publish(done(job_id))  # before anyone reads
        assert len(await asyncio.wait_for(collect(stream), 5)) == 1


async def test_other_jobs_are_not_delivered(events):
    mine, theirs = uuid4(), uuid4()
    async with events.subscribe(mine) as stream:
        await events.publish(processing(theirs, 0.45))
        await events.publish(done(theirs))
        await events.publish(done(mine))
        seen = await asyncio.wait_for(collect(stream), 5)
    assert [p.job_id for p in seen] == [mine]


async def test_leaving_the_block_drops_the_subscription(events):
    job_id = uuid4()
    probe = aioredis.from_url(os.environ["TEST_REDIS_URL"], protocol=2)
    try:
        async with events.subscribe(job_id):
            assert (await probe.pubsub_numsub(f"jobs:{job_id}"))[0][1] == 1
        await asyncio.sleep(0.1)
        assert (await probe.pubsub_numsub(f"jobs:{job_id}"))[0][1] == 0
    finally:
        await probe.aclose()


async def test_garbage_on_the_channel_does_not_break_the_stream(events):
    job_id = uuid4()
    probe = aioredis.from_url(os.environ["TEST_REDIS_URL"], protocol=2)
    try:
        async with events.subscribe(job_id) as stream:
            await probe.publish(f"jobs:{job_id}", "not json")
            await events.publish(done(job_id))
            seen = await asyncio.wait_for(collect(stream), 5)
    finally:
        await probe.aclose()
    assert [p.state for p in seen] == [JobState.DONE]
