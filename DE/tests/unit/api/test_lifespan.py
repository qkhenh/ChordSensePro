"""The app sweeps stale temporary directories at startup and keeps doing so in the background."""

import os
from uuid import uuid4

from src.api.app import create_app
from src.api.settings import ApiSettings
from src.ingestion.infrastructure.ffmpeg import FfmpegNormalizer
from tests.support.identity import FakeClock, make_container
from tests.support.ingestion import build_ingestion


async def test_startup_removes_stale_work_directories_and_shutdown_stops_the_sweeper(tmp_path):
    clock = FakeClock()
    container = make_container(clock)
    ingestion = build_ingestion(
        tmp_path / "data", quota=container.quota, clock=clock, normalizer=FfmpegNormalizer()
    )
    stale = tmp_path / "data" / "tmp" / str(uuid4())
    stale.mkdir(parents=True)
    (stale / "upload.bin").write_bytes(b"x")
    long_ago = clock().timestamp() - 10 * ingestion.settings.temp_ttl_s
    os.utime(stale, (long_ago, long_ago))

    app = create_app(container, ApiSettings(), ingestion=ingestion)
    async with app.router.lifespan_context(app):
        assert not stale.exists()
        assert not app.state.sweeper.done()
    assert app.state.sweeper.done()


async def test_an_app_without_ingestion_services_still_starts_and_stops():
    app = create_app(make_container(FakeClock()), ApiSettings())
    async with app.router.lifespan_context(app):
        assert not hasattr(app.state, "sweeper")
