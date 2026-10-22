"""contracts/openapi.json is generated from the Pydantic models; it must never drift."""

import json
from pathlib import Path

from src.analysis_contract.export import build_openapi_schema, render

COMMITTED = Path(__file__).parents[4] / "contracts" / "openapi.json"


def test_committed_openapi_matches_the_models():
    assert COMMITTED.read_text(encoding="utf-8") == render(), (
        "contracts/openapi.json is stale - run: rtk uv run python -m src.analysis_contract.export"
    )


def test_schema_exposes_the_types_the_frontend_needs():
    schemas = build_openapi_schema()["components"]["schemas"]
    for name in (
        "AnalysisResult",
        "ChordEvent",
        "Chart",
        "JobProgress",
        "PlaybackSource",
        "ErrorCode",
    ):
        assert name in schemas


def test_every_object_schema_forbids_unknown_fields():
    schemas = build_openapi_schema()["components"]["schemas"]
    for name, schema in schemas.items():
        if schema.get("type") == "object":
            assert schema.get("additionalProperties") is False, name


def test_output_is_deterministic():
    assert json.loads(render()) == build_openapi_schema()
    assert render() == render()
