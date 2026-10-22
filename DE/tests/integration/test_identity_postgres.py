"""Identity repositories against real PostgreSQL (database *_test)."""

import asyncio
import os
from datetime import UTC, date, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from src.identity.domain.errors import EmailTaken
from src.identity.domain.models import Role, User
from src.identity.infrastructure.postgres import PgUsageRepository, PgUserRepository

pytestmark = pytest.mark.integration

DE_DIR = Path(__file__).parents[2]


@pytest.fixture(scope="module", autouse=True)
def migrated():
    url = os.environ["TEST_DATABASE_URL"]
    assert url.rsplit("/", 1)[-1].endswith("_test"), "refusing to touch a non-test database"
    cfg = Config(str(DE_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(DE_DIR / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url)
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")


@pytest.fixture
async def engine() -> AsyncEngine:
    engine = create_async_engine(os.environ["TEST_DATABASE_URL"])
    async with engine.begin() as conn:
        await conn.exec_driver_sql("TRUNCATE app.users CASCADE")
    yield engine
    await engine.dispose()


def make_user(email="ann@example.com", role=Role.FREE) -> User:
    return User(
        id=uuid4(),
        email=email,
        name="Ann",
        password_hash="$2b$04$" + "x" * 53,
        role=role,
        is_active=True,
        created_at=datetime.now(UTC),
    )


class TestPgUserRepository:
    async def test_add_and_fetch_by_id_and_email(self, engine):
        repo = PgUserRepository(engine)
        user = await repo.add(make_user())
        assert await repo.get_by_id(user.id) == user
        assert (await repo.get_by_email("ANN@example.com")).id == user.id

    async def test_unknown_user_is_none(self, engine):
        repo = PgUserRepository(engine)
        assert await repo.get_by_id(uuid4()) is None
        assert await repo.get_by_email("nobody@example.com") is None

    async def test_duplicate_email_ignoring_case_raises_email_taken(self, engine):
        repo = PgUserRepository(engine)
        await repo.add(make_user("ann@example.com"))
        with pytest.raises(EmailTaken):
            await repo.add(make_user("ANN@Example.com"))

    async def test_set_role_and_active_flag_persist(self, engine):
        repo = PgUserRepository(engine)
        user = await repo.add(make_user())
        await repo.set_role(user.id, Role.PAID)
        await repo.set_active(user.id, False)
        stored = await repo.get_by_id(user.id)
        assert (stored.role, stored.is_active) == (Role.PAID, False)


class TestPgUsageRepository:
    DAY = date(2026, 10, 22)

    async def make(self, engine):
        user = await PgUserRepository(engine).add(make_user())
        return PgUsageRepository(engine), user.id

    async def test_consume_counts_up_to_the_limit_then_refuses(self, engine):
        usage, uid = await self.make(engine)
        results = [await usage.try_consume(uid, self.DAY, limit=3) for _ in range(5)]
        assert results == [1, 2, 3, None, None]
        assert await usage.used(uid, self.DAY) == 3

    async def test_concurrent_consumers_cannot_exceed_the_limit(self, engine):
        usage, uid = await self.make(engine)
        results = await asyncio.gather(
            *(usage.try_consume(uid, self.DAY, limit=5) for _ in range(20))
        )
        assert sum(r is not None for r in results) == 5
        assert await usage.used(uid, self.DAY) == 5

    async def test_days_are_independent(self, engine):
        usage, uid = await self.make(engine)
        await usage.try_consume(uid, self.DAY, limit=1)
        assert await usage.try_consume(uid, date(2026, 10, 23), limit=1) == 1

    async def test_refund_decrements_but_never_below_zero(self, engine):
        usage, uid = await self.make(engine)
        await usage.try_consume(uid, self.DAY, limit=5)
        await usage.refund(uid, self.DAY)
        await usage.refund(uid, self.DAY)
        assert await usage.used(uid, self.DAY) == 0

    async def test_used_is_zero_when_nothing_was_consumed(self, engine):
        usage, uid = await self.make(engine)
        assert await usage.used(uid, self.DAY) == 0
