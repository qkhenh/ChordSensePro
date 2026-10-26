import asyncio
from uuid import uuid4

import pytest

from src.analysis_contract.models import JobProgress, JobStage, JobState
from src.ingestion.infrastructure.events import InMemoryJobEvents


def processing(job_id, progress=0.45, stage=JobStage.SEPARATING) -> JobProgress:
    return JobProgress(job_id=job_id, state=JobState.PROCESSING, stage=stage, progress=progress)


def done(job_id) -> JobProgress:
    return JobProgress(job_id=job_id, state=JobState.DONE, progress=1.0, result_chart_id=uuid4())


async def collect(stream) -> list[JobProgress]:
    return [item async for item in stream]


async def test_a_subscriber_sees_its_job_until_it_finishes():
    events, job_id = InMemoryJobEvents(), uuid4()
    async with events.subscribe(job_id) as stream:
        await events.publish(processing(uuid4()))  # someone else's job
        await events.publish(processing(job_id, 0.45))
        await events.publish(done(job_id))
        seen = await asyncio.wait_for(collect(stream), 1)
    assert [p.state for p in seen] == [JobState.PROCESSING, JobState.DONE]
    assert all(p.job_id == job_id for p in seen)


async def test_registration_happens_on_entry_not_on_first_read():
    events, job_id = InMemoryJobEvents(), uuid4()
    async with events.subscribe(job_id) as stream:
        await events.publish(processing(job_id))  # before anyone iterates
        await events.publish(done(job_id))
        assert len(await asyncio.wait_for(collect(stream), 1)) == 2


async def test_every_subscriber_gets_every_event():
    events, job_id = InMemoryJobEvents(), uuid4()
    async with events.subscribe(job_id) as first, events.subscribe(job_id) as second:
        await events.publish(done(job_id))
        assert len(await collect(first)) == len(await collect(second)) == 1


async def test_leaving_unregisters_even_if_never_read():
    events, job_id = InMemoryJobEvents(), uuid4()
    async with events.subscribe(job_id):
        assert events.subscriber_count(job_id) == 1
    assert events.subscriber_count(job_id) == 0


async def test_an_error_inside_the_block_still_unregisters():
    events, job_id = InMemoryJobEvents(), uuid4()
    with pytest.raises(RuntimeError):
        async with events.subscribe(job_id):
            raise RuntimeError("client went away")
    assert events.subscriber_count(job_id) == 0


async def test_publishing_without_subscribers_is_harmless():
    await InMemoryJobEvents().publish(done(uuid4()))
