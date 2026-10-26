"""Shared pytest setup: load DE/.env and skip DB-backed tests when no test database."""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[1] / ".env")


def pytest_collection_modifyitems(config, items):
    needs = {
        "integration": ("TEST_DATABASE_URL", "needs PostgreSQL"),
        "redis": ("TEST_REDIS_URL", "needs Redis"),
    }
    for marker, (variable, why) in needs.items():
        if os.environ.get(variable):
            continue
        skip = pytest.mark.skip(reason=f"{variable} not set ({why})")
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)
