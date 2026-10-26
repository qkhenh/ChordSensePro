"""HTTP gateway settings (env prefix ``API_``, also read from DE/.env)."""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.identity.settings import ENV_FILE


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="API_", env_file=ENV_FILE, extra="ignore")

    # Vite dev server (npm run dev / preview). A JSON list when set through the environment.
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://localhost:4173"]
    )

    @field_validator("cors_origins")
    @classmethod
    def _no_wildcard(cls, value: list[str]) -> list[str]:
        if "*" in value:
            raise ValueError("wildcard CORS origin is not allowed: list the origins explicitly")
        return value
