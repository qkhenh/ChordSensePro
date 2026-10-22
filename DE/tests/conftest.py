"""Shared pytest setup: load DE/.env and skip DB-backed tests when no test database."""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[1] / ".env")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("TEST_DATABASE_URL"):
        return
    skip = pytest.mark.skip(reason="TEST_DATABASE_URL not set (needs PostgreSQL)")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
