"""Ingestion settings (env prefix ``INGEST_``, also read from DE/.env)."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.identity.settings import ENV_FILE

DE_DIR = Path(__file__).parents[2]


class IngestSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INGEST_", env_file=ENV_FILE, extra="ignore")

    data_dir: Path = DE_DIR / "data"
    # NFR-12. "100 MB" is taken as 100 MiB (ruling D6 in the M2 plan).
    max_upload_bytes: int = Field(default=100 * 1024 * 1024, gt=0)
    max_duration_ms: int = Field(default=600_000, gt=0)
    job_timeout_s: float = Field(default=90, gt=0)  # UC1 EF1
    mock_stage_delay_ms: int = Field(default=0, ge=0)
    engine: Literal["mock"] = "mock"
    media_resolver: Literal["fake"] = "fake"
    job_runner: Literal["inprocess", "celery"] = "inprocess"
    temp_ttl_s: int = Field(default=3600, gt=0)
    redis_url: str = "redis://127.0.0.1:6379/0"  # Celery broker and job events
