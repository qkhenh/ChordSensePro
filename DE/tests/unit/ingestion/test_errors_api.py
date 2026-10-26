import httpx
import pytest
from fastapi import FastAPI

from src.analysis_contract.models import ERROR_HTTP_STATUS, ErrorCode
from src.api.errors import install_error_handlers
from src.ingestion.domain.errors import IngestionError


def app_raising(code: ErrorCode) -> FastAPI:
    app = FastAPI()
    install_error_handlers(app)

    @app.get("/boom")
    async def boom():
        raise IngestionError(code, f"because {code.value}")

    return app


@pytest.mark.parametrize("code", list(ErrorCode))
async def test_each_error_code_maps_to_its_contract_status(code):
    transport = httpx.ASGITransport(app=app_raising(code))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/boom")
    assert r.status_code == ERROR_HTTP_STATUS[code]
    assert r.json() == {"code": code.value, "message": f"because {code.value}"}


def test_ingestion_error_carries_retryable_and_defaults_to_false():
    assert IngestionError(ErrorCode.AI_FAILURE, "x").retryable is False
    assert IngestionError(ErrorCode.AI_FAILURE, "x", retryable=True).retryable is True
