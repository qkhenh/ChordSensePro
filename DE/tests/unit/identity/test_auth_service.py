import pytest

from src.identity.domain.errors import (
    AccountDisabled,
    EmailTaken,
    InvalidCredentials,
    InvalidToken,
    TokenRevoked,
    WeakPassword,
)
from src.identity.domain.models import Role
from tests.support.identity import PASSWORD, make_container


@pytest.fixture
def c():
    return make_container()


async def test_register_creates_a_free_active_user_with_a_hashed_password(c):
    user = await c.auth.register("Ann@Example.com", PASSWORD, name="Ann")
    assert user.role is Role.FREE
    assert user.is_active is True
    assert user.email == "ann@example.com"
    assert user.password_hash != PASSWORD
    assert user.password_hash.startswith("$2")


async def test_duplicate_email_is_rejected_ignoring_case_and_spaces(c):
    await c.auth.register("ann@example.com", PASSWORD)
    with pytest.raises(EmailTaken):
        await c.auth.register("  ANN@example.com ", PASSWORD)


@pytest.mark.parametrize(
    "password",
    ["", "short", "       ", " " * 12, "a" * 73],
)
async def test_weak_or_unhashable_passwords_are_rejected(c, password):
    with pytest.raises(WeakPassword):
        await c.auth.register("w@example.com", password)


async def test_password_of_exactly_eight_characters_is_accepted(c):
    assert await c.auth.register("ok@example.com", "abcd1234")


async def test_login_returns_the_user_and_a_working_token_pair(c):
    registered = await c.auth.register("ann@example.com", PASSWORD)
    user, pair = await c.auth.login("ANN@example.com", PASSWORD)
    assert user.id == registered.id
    claims = await c.tokens.verify(pair.access_token, "access")
    assert claims.user_id == user.id


async def test_wrong_password_and_unknown_email_look_identical(c):
    await c.auth.register("ann@example.com", PASSWORD)
    with pytest.raises(InvalidCredentials) as wrong:
        await c.auth.login("ann@example.com", "not-the-password")
    with pytest.raises(InvalidCredentials) as unknown:
        await c.auth.login("nobody@example.com", PASSWORD)
    assert str(wrong.value) == str(unknown.value)


async def test_unknown_email_still_pays_the_hashing_cost(c, monkeypatch):
    calls = []
    real = c.auth._hasher.verify
    monkeypatch.setattr(c.auth._hasher, "verify", lambda p, h: calls.append(h) or real(p, h))
    with pytest.raises(InvalidCredentials):
        await c.auth.login("nobody@example.com", PASSWORD)
    assert len(calls) == 1  # no early return that would reveal which emails exist


async def test_disabled_account_cannot_log_in_even_with_the_right_password(c):
    user = await c.auth.register("ann@example.com", PASSWORD)
    await c.users.set_active(user.id, False)
    with pytest.raises(AccountDisabled):
        await c.auth.login("ann@example.com", PASSWORD)


async def test_refresh_rotates_the_pair_and_burns_the_old_refresh_token(c):
    await c.auth.register("ann@example.com", PASSWORD)
    _, first = await c.auth.login("ann@example.com", PASSWORD)
    _, second = await c.auth.refresh(first.refresh_token)
    assert second.refresh_token != first.refresh_token
    with pytest.raises(TokenRevoked):
        await c.auth.refresh(first.refresh_token)  # replay of a used refresh token


async def test_an_access_token_cannot_be_used_to_refresh(c):
    await c.auth.register("ann@example.com", PASSWORD)
    _, pair = await c.auth.login("ann@example.com", PASSWORD)
    with pytest.raises(InvalidToken):
        await c.auth.refresh(pair.access_token)


async def test_refresh_fails_once_the_account_is_disabled(c):
    user = await c.auth.register("ann@example.com", PASSWORD)
    _, pair = await c.auth.login("ann@example.com", PASSWORD)
    await c.users.set_active(user.id, False)
    with pytest.raises(AccountDisabled):
        await c.auth.refresh(pair.refresh_token)


async def test_logout_revokes_the_refresh_token(c):
    await c.auth.register("ann@example.com", PASSWORD)
    _, pair = await c.auth.login("ann@example.com", PASSWORD)
    await c.auth.logout(pair.refresh_token)
    with pytest.raises(TokenRevoked):
        await c.auth.refresh(pair.refresh_token)


@pytest.mark.parametrize("token", ["", "garbage", "a.b.c"])
async def test_logout_with_a_bad_token_is_silently_ok(c, token):
    await c.auth.logout(token)  # signing out must never fail or reveal anything


async def test_logout_twice_is_fine(c):
    await c.auth.register("ann@example.com", PASSWORD)
    _, pair = await c.auth.login("ann@example.com", PASSWORD)
    await c.auth.logout(pair.refresh_token)
    await c.auth.logout(pair.refresh_token)
