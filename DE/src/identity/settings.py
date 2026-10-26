"""Authentication settings (env prefix ``AUTH_``, also read from DE/.env)."""

from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path(__file__).parents[2] / ".env"
MIN_BCRYPT_COST = 12  # NFR-09
MIN_SECRET_LENGTH = 32


class AuthSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AUTH_", env_file=ENV_FILE, extra="ignore")

    jwt_secret: SecretStr
    access_ttl_minutes: int = Field(default=15, gt=0)
    refresh_ttl_days: int = Field(default=14, gt=0)
    bcrypt_cost: int = MIN_BCRYPT_COST
    media_ticket_ttl_minutes: int = Field(default=30, gt=0)

    @field_validator("jwt_secret")
    @classmethod
    def _secret_is_long_enough(cls, value: SecretStr) -> SecretStr:
        if len(value.get_secret_value()) < MIN_SECRET_LENGTH:
            raise ValueError(f"jwt_secret must be at least {MIN_SECRET_LENGTH} characters")
        return value

    @field_validator("bcrypt_cost")
    @classmethod
    def _cost_is_strong_enough(cls, value: int) -> int:
        if value < MIN_BCRYPT_COST:
            raise ValueError(f"bcrypt_cost must be >= {MIN_BCRYPT_COST}")
        return value
