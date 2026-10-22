import pytest

from src.persistence import database_url
from src.shared.infrastructure.settings.config import get_settings


@pytest.fixture(autouse=True)
def fresh_settings():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_database_url_env_var_wins(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@h:1/db")
    assert database_url() == "postgresql+asyncpg://u:p@h:1/db"


def test_database_url_is_built_from_postgres_variables(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    for key, value in {
        "POSTGRES_HOST": "db.local",
        "POSTGRES_PORT": "6543",
        "POSTGRES_DB": "cs",
        "POSTGRES_USER": "me",
        "POSTGRES_PASSWORD": "secret",
    }.items():
        monkeypatch.setenv(key, value)
    assert database_url() == "postgresql+asyncpg://me:secret@db.local:6543/cs"


def test_database_url_reads_the_de_dotenv_file(monkeypatch, tmp_path):
    # The nested PostgresSettings ignores .env on its own, so database_url() must load it.
    env_file = tmp_path / ".env"
    env_file.write_text(
        "POSTGRES_USER=from_file\nPOSTGRES_PASSWORD=pw\nPOSTGRES_DB=filedb\n", encoding="utf-8"
    )
    for key in ("DATABASE_URL", "POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("src.persistence.ENV_FILE", env_file)
    assert "from_file:pw@" in database_url()
    assert database_url().endswith("/filedb")
