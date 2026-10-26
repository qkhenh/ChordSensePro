from types import SimpleNamespace

import httpx
import pytest

from src.api.app import create_app
from src.api.settings import ApiSettings
from src.ingestion.infrastructure.ffmpeg import FfmpegNormalizer
from tests.support.api import make_app
from tests.support.identity import FakeClock, make_container
from tests.support.ingestion import build_ingestion


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def container(clock):
    return make_container(clock)


@pytest.fixture
async def client(container):
    transport = httpx.ASGITransport(app=make_app(container))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
async def make_api(tmp_path, clock):
    """Factory for a full app (identity + ingestion, real ffmpeg) with an HTTP client."""
    opened: list[tuple[httpx.AsyncClient, object]] = []

    async def factory(*, free_quota: int = 5, max_upload_bytes: int | None = None, engine=None):
        container = make_container(clock, free_quota=free_quota)
        data_dir = tmp_path / "data"
        ingestion = build_ingestion(
            data_dir,
            quota=container.quota,
            clock=clock,
            normalizer=FfmpegNormalizer(),
            max_upload_bytes=max_upload_bytes,
            engine=engine,
        )
        app = create_app(container, ApiSettings(), ingestion=ingestion)
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        http = httpx.AsyncClient(transport=transport, base_url="http://test")
        opened.append((http, ingestion))
        return SimpleNamespace(
            client=http, ingestion=ingestion, container=container, data_dir=data_dir, app=app
        )

    yield factory

    for http, ingestion in opened:  # let background jobs finish before the loop closes
        await ingestion.runner.drain()
        await http.aclose()
