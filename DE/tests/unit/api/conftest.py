import httpx
import pytest

from tests.support.api import make_app
from tests.support.identity import FakeClock, make_container


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
