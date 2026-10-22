"""NFR-10: automated role x endpoint matrix - zero unauthorized successes."""

from uuid import UUID

import pytest

from src.identity.domain.models import PERSISTED_ROLES, PRO_ROLES, Role
from tests.support.api import ADMIN, ANY, CONTENT, PRO
from tests.support.identity import PASSWORD

ALLOWED = {
    ANY: set(PERSISTED_ROLES),
    PRO: set(PRO_ROLES),
    ADMIN: {Role.IT_ADMIN},
    CONTENT: {Role.MUSIC_PM, Role.EXPERT},
}
CASES = [(role, path) for path in ALLOWED for role in sorted(PERSISTED_ROLES)]


async def token_for(client, container, role: Role) -> str:
    email = f"{role.value}@example.com"
    body = (
        await client.post("/api/v1/auth/register", json={"email": email, "password": PASSWORD})
    ).json()
    await container.users.set_role(UUID(body["user"]["id"]), role)
    return body["access_token"]


@pytest.mark.parametrize(("role", "path"), CASES, ids=[f"{r.value}->{p}" for r, p in CASES])
async def test_role_can_only_reach_what_it_is_allowed(client, container, role, path):
    token = await token_for(client, container, role)
    response = await client.get(path, headers={"Authorization": f"Bearer {token}"})
    expected = 200 if role in ALLOWED[path] else 403
    assert response.status_code == expected, response.text
    if expected == 403:
        assert response.json()["code"] == "forbidden"


@pytest.mark.parametrize("path", list(ALLOWED))
async def test_guests_never_get_in(client, path):
    assert (await client.get(path)).status_code == 401


def test_guest_is_not_a_persisted_role():
    assert Role.GUEST not in PERSISTED_ROLES
    assert len(PERSISTED_ROLES) == 7
    assert len(Role) == 8  # NFR-10 names eight roles
