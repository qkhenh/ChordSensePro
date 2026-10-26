from datetime import UTC, date, datetime

from src.analysis_contract.models import JobState
from tests.support.ingestion import make_world

DAY_ONE = date(2026, 10, 26)
DAY_TWO = date(2026, 10, 27)


async def test_a_refund_goes_back_to_the_day_that_was_charged(tmp_path):
    world = make_world(tmp_path)
    world.clock.now = datetime(2026, 10, 26, 23, 59, 59, tzinfo=UTC)
    job = await world.submit(name="fail-ai.mp3")
    assert job.charged_on == DAY_ONE
    assert await world.usage.used(world.user.id, DAY_ONE) == 1

    world.clock.advance(seconds=30)  # the job fails after midnight UTC
    await world.executor.run(job.id)

    assert (await world.jobs.get(job.id)).state == JobState.FAILED
    assert await world.usage.used(world.user.id, DAY_ONE) == 0
    assert await world.usage.used(world.user.id, DAY_TWO) == 0


async def test_a_job_charged_after_midnight_is_refunded_to_the_new_day(tmp_path):
    world = make_world(tmp_path)
    world.clock.now = datetime(2026, 10, 27, 0, 0, 1, tzinfo=UTC)
    job = await world.submit(name="fail-ai.mp3")
    await world.executor.run(job.id)
    assert await world.usage.used(world.user.id, DAY_TWO) == 0
