"""Gateway persistence: table definitions and database URL resolution."""

import os
from pathlib import Path

from dotenv import load_dotenv

from src.shared.infrastructure.settings.config import get_settings

ENV_FILE = Path(__file__).parents[2] / ".env"


def database_url() -> str:
    """``DATABASE_URL`` wins, otherwise the POSTGRES_* settings (asyncpg driver).

    DE/.env is loaded first (without overriding real environment variables): the nested
    ``PostgresSettings`` does not read it by itself, only the top-level ``Settings`` does.
    """
    load_dotenv(ENV_FILE, override=False)
    return os.environ.get("DATABASE_URL") or get_settings().postgres.url
