"""Generate ``contracts/openapi.json`` (components only) from the Pydantic models.

rtk uv run python -m src.analysis_contract.export
"""

import json
from pathlib import Path

from pydantic.json_schema import models_json_schema

from src.api import schemas as api_schemas

from . import models

OUTPUT = Path(__file__).parents[3] / "contracts" / "openapi.json"

_EXPORTED = (
    models.AnalysisRequest,
    models.AnalysisResult,
    models.JobProgress,
    models.Chart,
    models.EventOverride,
    models.ChordEvent,
    models.PlaybackSource,
    api_schemas.RegisterRequest,
    api_schemas.LoginRequest,
    api_schemas.RefreshRequest,
    api_schemas.UserOut,
    api_schemas.AuthResponse,
    api_schemas.QuotaOut,
    api_schemas.JobAccepted,
    api_schemas.AnalyzeUrlRequest,
)


def build_openapi_schema() -> dict:
    _, defs = models_json_schema(
        [(m, "validation") for m in _EXPORTED], ref_template="#/components/schemas/{model}"
    )
    return {
        "openapi": "3.1.0",
        "info": {"title": "ChordSense Pro analysis contract", "version": models.SCHEMA_VERSION},
        "paths": {},
        "components": {"schemas": defs["$defs"]},
    }


def render() -> str:
    return json.dumps(build_openapi_schema(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(render(), encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
