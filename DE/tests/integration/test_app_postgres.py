"""``create_app()`` with no injected container: wires PostgreSQL repositories from the env."""

import os
from uuid import uuid4

import httpx
import pytest
from alembic import command
from alembic.config import Config

from src.api.app import create_app
from tests.support.identity import PASSWORD

pytestmark = pytest.mark.integration

DE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module", autouse=True)
def migrated():
    url = os.environ["TEST_DATABASE_URL"]
    assert url.rsplit("/", 1)[-1].endswith("_test"), "refusing to touch a non-test database"
    cfg = Config(os.path.join(DE_DIR, "alembic.ini"))
    cfg.set_main_option("script_location", os.path.join(DE_DIR, "migrations"))
    cfg.set_main_option("sqlalchemy.url", url)
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")


@pytest.fixture
async def client(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", os.environ["TEST_DATABASE_URL"])
    app = create_app()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


async def test_register_login_me_and_quota_hit_the_real_database(client):
    email = f"{uuid4().hex[:8]}@example.com"
    r = await client.post(
        "/api/v1/auth/register", json={"email": email, "password": PASSWORD, "name": "Ann"}
    )
    assert r.status_code == 201

    login = await client.post("/api/v1/auth/login", json={"email": email, "password": PASSWORD})
    assert login.status_code == 200
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    assert (await client.get("/api/v1/users/me", headers=headers)).json()["email"] == email
    quota = (await client.get("/api/v1/users/me/quota", headers=headers)).json()
    assert quota["limit"] == 5

    dup = await client.post(
        "/api/v1/auth/register", json={"email": email.upper(), "password": PASSWORD, "name": "x"}
    )
    assert dup.status_code == 409


async def test_lifespan_disposes_the_engine_on_shutdown(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", os.environ["TEST_DATABASE_URL"])
    app = create_app()
    async with app.router.lifespan_context(app):
        engine = app.state.engine
    # dispose() swaps in a fresh pool, so nothing is left checked out or idle.
    assert engine.pool.checkedin() == 0
    assert engine.pool.checkedout() == 0
