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


async def test_an_uploaded_wav_becomes_a_chart_in_the_real_database(monkeypatch, tmp_path):
    from tests.support.audio import exact_size_wav

    monkeypatch.setenv("DATABASE_URL", os.environ["TEST_DATABASE_URL"])
    monkeypatch.setenv("INGEST_DATA_DIR", str(tmp_path))
    app = create_app()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            email = f"{uuid4().hex[:8]}@example.com"
            r = await client.post(
                "/api/v1/auth/register", json={"email": email, "password": PASSWORD, "name": "Ann"}
            )
            headers = {"Authorization": f"Bearer {r.json()['access_token']}"}

            up = await client.post(
                "/api/v1/analyze/file",
                headers=headers,
                files={"file": ("Real Song.wav", exact_size_wav(20_044), "audio/wav")},
            )
            assert up.status_code == 202
            await app.state.ingestion.runner.drain()

            job_id = up.json()["job_id"]
            status = (await client.get(f"/api/v1/analyze/status/{job_id}", headers=headers)).json()
            assert status["state"] == "done"
            chart = await client.get(f"/api/v1/charts/{status['result_chart_id']}", headers=headers)
            assert chart.status_code == 200 and chart.json()["title"] == "Real Song"
            quota = (await client.get("/api/v1/users/me/quota", headers=headers)).json()
            assert quota["used"] == 1


async def test_the_celery_worker_builds_an_executor_from_the_environment(monkeypatch, tmp_path):
    from src.ingestion.application.executor import JobExecutor
    from src.ingestion.worker import open_executor

    monkeypatch.setenv("DATABASE_URL", os.environ["TEST_DATABASE_URL"])
    monkeypatch.setenv("INGEST_DATA_DIR", str(tmp_path))
    async with open_executor() as executor:
        assert isinstance(executor, JobExecutor)
        assert await executor.recover_orphans() == 0  # it can really reach the database
