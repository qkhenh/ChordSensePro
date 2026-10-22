from datetime import UTC, datetime

import pytest

from src.identity.application.quota import tier_for
from src.identity.domain.errors import QuotaExceeded
from src.identity.domain.models import Role
from tests.support.identity import PASSWORD, FakeClock, make_container


async def user_with_role(c, role: Role, email="u@example.com"):
    user = await c.auth.register(email, PASSWORD)
    await c.users.set_role(user.id, role)
    return await c.users.get_by_id(user.id)


async def test_free_user_starts_with_the_policy_limit_of_five():
    c = make_container()
    user = await user_with_role(c, Role.FREE)
    status = await c.quota.status(user)
    assert (status.tier, status.limit, status.used, status.remaining) == ("free", 5, 0, 5)


async def test_sixth_analysis_of_the_day_is_rejected_with_the_reset_time():
    c = make_container()
    user = await user_with_role(c, Role.FREE)
    for _ in range(5):
        await c.quota.consume(user)
    with pytest.raises(QuotaExceeded) as exc:
        await c.quota.consume(user)
    assert exc.value.limit == 5
    assert exc.value.reset_at == datetime(2026, 10, 23, 0, 0, tzinfo=UTC)
    assert (await c.quota.status(user)).remaining == 0


async def test_quota_resets_at_midnight_utc():
    clock = FakeClock(datetime(2026, 10, 22, 23, 59, tzinfo=UTC))
    c = make_container(clock)
    user = await user_with_role(c, Role.FREE)
    for _ in range(5):
        await c.quota.consume(user)
    clock.advance(minutes=1)  # 00:00 next day
    status = await c.quota.status(user)
    assert (status.used, status.remaining) == (0, 5)
    await c.quota.consume(user)


async def test_reset_time_is_always_the_next_midnight():
    clock = FakeClock(datetime(2026, 10, 22, 0, 0, tzinfo=UTC))
    c = make_container(clock)
    user = await user_with_role(c, Role.FREE)
    assert (await c.quota.status(user)).resets_at == datetime(2026, 10, 23, 0, 0, tzinfo=UTC)


async def test_refund_gives_the_slot_back_to_the_day_it_was_charged():
    clock = FakeClock(datetime(2026, 10, 22, 23, 59, tzinfo=UTC))
    c = make_container(clock)
    user = await user_with_role(c, Role.FREE)
    receipt = await c.quota.consume(user)
    clock.advance(minutes=2)  # the job failed after midnight
    await c.quota.refund(receipt)
    clock.advance(minutes=-2)
    assert (await c.quota.status(user)).used == 0


async def test_refund_never_takes_the_counter_below_zero():
    c = make_container()
    user = await user_with_role(c, Role.FREE)
    receipt = await c.quota.consume(user)
    await c.quota.refund(receipt)
    await c.quota.refund(receipt)
    assert (await c.quota.status(user)).used == 0


@pytest.mark.parametrize(
    "role", [Role.PAID, Role.TEACHER, Role.EXPERT, Role.MUSIC_PM, Role.IT_ADMIN]
)
async def test_pro_and_staff_roles_are_unlimited(role):
    c = make_container()
    user = await user_with_role(c, role)
    for _ in range(50):
        await c.quota.consume(user)
    status = await c.quota.status(user)
    assert (status.limit, status.remaining) == (None, None)
    assert status.tier == "pro"


async def test_limit_is_read_from_the_policy_not_hard_coded():
    c = make_container(free_quota=2)
    user = await user_with_role(c, Role.FREE)
    await c.quota.consume(user)
    await c.quota.consume(user)
    with pytest.raises(QuotaExceeded):
        await c.quota.consume(user)


async def test_a_zero_quota_policy_blocks_everything():
    c = make_container(free_quota=0)
    user = await user_with_role(c, Role.FREE)
    with pytest.raises(QuotaExceeded):
        await c.quota.consume(user)


async def test_users_do_not_share_counters():
    c = make_container()
    a = await user_with_role(c, Role.FREE, "a@example.com")
    b = await user_with_role(c, Role.FREE, "b@example.com")
    for _ in range(5):
        await c.quota.consume(a)
    assert (await c.quota.status(b)).used == 0


def test_students_count_as_the_education_tier():
    assert tier_for(Role.STUDENT) == "education"
    assert tier_for(Role.FREE) == "free"
    assert tier_for(Role.PAID) == "pro"
