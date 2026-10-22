import pytest

from tests.support.api import ANY, CONSUME
from tests.support.identity import PASSWORD

REGISTER = "/api/v1/auth/register"
LOGIN = "/api/v1/auth/login"
REFRESH = "/api/v1/auth/refresh"
ME = "/api/v1/users/me"
QUOTA = "/api/v1/users/me/quota"


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


async def register(client, email="ann@example.com", password=PASSWORD, **extra):
    return await client.post(REGISTER, json={"email": email, "password": password, **extra})


async def tokens(client, email="ann@example.com"):
    return (await register(client, email)).json()


class TestRegister:
    async def test_returns_201_with_user_and_tokens_and_never_the_password(self, client):
        r = await register(client, name="Ann")
        assert r.status_code == 201
        body = r.json()
        assert body["user"]["email"] == "ann@example.com"
        assert body["user"]["role"] == "free"
        assert body["user"]["name"] == "Ann"
        assert body["token_type"] == "bearer"
        assert body["expires_in"] == 900
        assert body["access_token"] and body["refresh_token"]
        assert "password" not in r.text.lower()

    async def test_duplicate_email_is_409_with_a_stable_error_code(self, client):
        await register(client)
        r = await register(client, email="ANN@example.com")
        assert r.status_code == 409
        assert r.json()["code"] == "email_taken"
        assert r.json()["message"]

    async def test_malformed_email_is_422(self, client):
        assert (await register(client, email="not-an-email")).status_code == 422

    async def test_weak_password_is_422_with_its_own_code(self, client):
        r = await register(client, password="short")
        assert r.status_code == 422
        assert r.json()["code"] == "weak_password"


class TestLogin:
    async def test_success(self, client):
        await register(client)
        r = await client.post(LOGIN, json={"email": "Ann@Example.com", "password": PASSWORD})
        assert r.status_code == 200
        assert r.json()["user"]["email"] == "ann@example.com"

    async def test_wrong_password_is_401_and_asks_for_bearer_auth(self, client):
        await register(client)
        r = await client.post(
            LOGIN, json={"email": "ann@example.com", "password": "nope-nope-nope"}
        )
        assert r.status_code == 401
        assert r.json()["code"] == "invalid_credentials"
        assert r.headers["www-authenticate"] == "Bearer"

    async def test_disabled_account_is_403(self, client, container):
        body = await tokens(client)
        await container.users.set_active(
            (await container.users.get_by_email("ann@example.com")).id, False
        )
        r = await client.post(LOGIN, json={"email": "ann@example.com", "password": PASSWORD})
        assert r.status_code == 403
        assert r.json()["code"] == "account_disabled"
        assert body


class TestRefresh:
    async def test_rotates_tokens_and_rejects_replay(self, client):
        first = await tokens(client)
        r = await client.post(REFRESH, json={"refresh_token": first["refresh_token"]})
        assert r.status_code == 200
        assert r.json()["refresh_token"] != first["refresh_token"]
        replay = await client.post(REFRESH, json={"refresh_token": first["refresh_token"]})
        assert replay.status_code == 401
        assert replay.json()["code"] == "token_revoked"

    async def test_access_token_is_not_accepted_as_a_refresh_token(self, client):
        first = await tokens(client)
        r = await client.post(REFRESH, json={"refresh_token": first["access_token"]})
        assert r.status_code == 401
        assert r.json()["code"] == "invalid_token"


class TestLogout:
    LOGOUT = "/api/v1/auth/logout"

    async def test_revokes_the_refresh_token_and_answers_204(self, client):
        body = await tokens(client)
        r = await client.post(self.LOGOUT, json={"refresh_token": body["refresh_token"]})
        assert r.status_code == 204
        assert r.content == b""
        replay = await client.post(REFRESH, json={"refresh_token": body["refresh_token"]})
        assert replay.status_code == 401
        assert replay.json()["code"] == "token_revoked"

    async def test_an_unknown_token_is_still_204(self, client):
        r = await client.post(self.LOGOUT, json={"refresh_token": "garbage"})
        assert r.status_code == 204


class TestAuthenticatedAccess:
    async def test_me_returns_the_current_user(self, client):
        body = await tokens(client)
        r = await client.get(ME, headers=bearer(body["access_token"]))
        assert r.status_code == 200
        assert r.json()["email"] == "ann@example.com"
        assert "password_hash" not in r.text

    async def test_missing_credentials_are_401_not_authenticated(self, client):
        r = await client.get(ME)
        assert r.status_code == 401
        assert r.json()["code"] == "not_authenticated"
        assert r.headers["www-authenticate"] == "Bearer"

    @pytest.mark.parametrize(
        "header", ["Basic abc", "Bearer", "Bearer ", "token abc", "Bearer a b"]
    )
    async def test_malformed_authorization_header_is_401(self, client, header):
        r = await client.get(ME, headers={"Authorization": header})
        assert r.status_code == 401

    async def test_garbage_token_is_401_invalid_token(self, client):
        r = await client.get(ME, headers=bearer("garbage.token.value"))
        assert r.status_code == 401
        assert r.json()["code"] == "invalid_token"

    async def test_expired_token_is_401_token_expired(self, client, clock):
        body = await tokens(client)
        clock.advance(minutes=16)
        r = await client.get(ME, headers=bearer(body["access_token"]))
        assert r.status_code == 401
        assert r.json()["code"] == "token_expired"

    async def test_refresh_token_cannot_authenticate_requests(self, client):
        body = await tokens(client)
        assert (await client.get(ME, headers=bearer(body["refresh_token"]))).status_code == 401

    async def test_token_of_a_deactivated_user_stops_working_immediately(self, client, container):
        body = await tokens(client)
        user = await container.users.get_by_email("ann@example.com")
        await container.users.set_active(user.id, False)
        assert (await client.get(ME, headers=bearer(body["access_token"]))).status_code == 401


class TestQuotaEndpoint:
    async def test_reports_tier_limit_usage_and_reset_time(self, client):
        body = await tokens(client)
        r = await client.get(QUOTA, headers=bearer(body["access_token"]))
        assert r.status_code == 200
        assert r.json() == {
            "tier": "free",
            "limit": 5,
            "used": 0,
            "remaining": 5,
            "resets_at": "2026-10-23T00:00:00Z",
        }

    async def test_usage_is_reflected(self, client):
        body = await tokens(client)
        h = bearer(body["access_token"])
        for _ in range(2):
            assert (await client.post(CONSUME, headers=h)).status_code == 200
        assert (await client.get(QUOTA, headers=h)).json()["remaining"] == 3

    async def test_exhausted_quota_is_429_with_retry_after(self, client):
        h = bearer((await tokens(client))["access_token"])
        for _ in range(5):
            await client.post(CONSUME, headers=h)
        r = await client.post(CONSUME, headers=h)
        assert r.status_code == 429
        assert r.json()["code"] == "quota_exceeded"
        assert r.json()["resets_at"] == "2026-10-23T00:00:00Z"
        assert int(r.headers["retry-after"]) == 12 * 3600  # clock is 12:00 UTC

    async def test_requires_authentication(self, client):
        assert (await client.get(QUOTA)).status_code == 401


async def test_any_route_works_with_a_valid_token(client):
    body = await tokens(client)
    assert (await client.get(ANY, headers=bearer(body["access_token"]))).status_code == 200
