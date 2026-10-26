"""Concurrent uploads at the daily limit, against real PostgreSQL."""

import asyncio
import os
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from alembic import command
from alembic.config import Config

from src.api.app import create_app
from tests.support.audio import exact_size_wav
from tests.support.identity import PASSWORD

pytestmark = pytest.mark.integration

DE_DIR = Path(__file__).parents[2]


@pytest.fixture(scope="module", autouse=True)
def migrated():
    url = os.environ["TEST_DATABASE_URL"]
    assert url.rsplit("/", 1)[-1].endswith("_test"), "refusing to touch a non-test database"
    cfg = Config(str(DE_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(DE_DIR / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url)
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")


async def test_ten_simultaneous_uploads_never_exceed_the_daily_limit(monkeypatch, tmp_path):
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

            async def upload(n: int) -> int:
                files = {"file": (f"s{n}.wav", exact_size_wav(8044, fill=n), "audio/wav")}
                return (
                    await client.post("/api/v1/analyze/file", headers=headers, files=files)
                ).status_code

            assert [await upload(n) for n in (1, 2)] == [202, 202]  # 3 of the 5 left
            outcomes = await asyncio.gather(*(upload(n) for n in range(10, 20)))
            assert sorted(outcomes) == [202] * 3 + [429] * 7

            await app.state.ingestion.runner.drain()
            quota = (await client.get("/api/v1/users/me/quota", headers=headers)).json()
            assert (quota["used"], quota["remaining"]) == (5, 0)
