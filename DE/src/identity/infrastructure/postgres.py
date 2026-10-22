"""PostgreSQL adapters (SQLAlchemy Core, asyncpg) for the identity ports."""

from datetime import date
from uuid import UUID

from sqlalchemy import func, select, text, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine

from src.identity.domain.errors import EmailTaken
from src.identity.domain.models import Role, User
from src.persistence.tables import usage_counters, users


def _to_user(row) -> User:
    return User(
        id=row.id,
        email=row.email,
        name=row.name,
        password_hash=row.password_hash,
        role=Role(row.role),
        is_active=row.is_active,
        created_at=row.created_at,
    )


class PgUserRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def add(self, user: User) -> User:
        stmt = (
            insert(users)
            .values(
                id=user.id,
                email=user.email,
                name=user.name,
                password_hash=user.password_hash,
                role=user.role.value,
                is_active=user.is_active,
                created_at=user.created_at,
                updated_at=user.created_at,
            )
            # Conflict target is the case-insensitive unique index ux_users_email_lower.
            .on_conflict_do_nothing(index_elements=[text("lower(email)")])
            .returning(users.c.id)
        )
        async with self._engine.begin() as conn:
            if (await conn.execute(stmt)).first() is None:
                raise EmailTaken
        return user

    async def get_by_id(self, user_id: UUID) -> User | None:
        async with self._engine.connect() as conn:
            row = (await conn.execute(select(users).where(users.c.id == user_id))).first()
        return _to_user(row) if row else None

    async def get_by_email(self, email: str) -> User | None:
        stmt = select(users).where(func.lower(users.c.email) == email.strip().lower())
        async with self._engine.connect() as conn:
            row = (await conn.execute(stmt)).first()
        return _to_user(row) if row else None

    async def _update(self, user_id: UUID, **values) -> None:
        stmt = update(users).where(users.c.id == user_id).values(updated_at=func.now(), **values)
        async with self._engine.begin() as conn:
            await conn.execute(stmt)

    async def set_role(self, user_id: UUID, role: Role) -> None:
        await self._update(user_id, role=role.value)

    async def set_active(self, user_id: UUID, active: bool) -> None:
        await self._update(user_id, is_active=active)


class PgUsageRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def try_consume(self, user_id: UUID, day: date, limit: int) -> int | None:
        """One atomic statement: concurrent callers can never exceed ``limit``."""
        stmt = (
            insert(usage_counters)
            .values(user_id=user_id, period_date=day, analyses_used=1)
            .on_conflict_do_update(
                index_elements=[usage_counters.c.user_id, usage_counters.c.period_date],
                set_={"analyses_used": usage_counters.c.analyses_used + 1},
                where=usage_counters.c.analyses_used < limit,
            )
            .returning(usage_counters.c.analyses_used)
        )
        async with self._engine.begin() as conn:
            row = (await conn.execute(stmt)).first()
        return row[0] if row else None

    async def refund(self, user_id: UUID, day: date) -> None:
        stmt = (
            update(usage_counters)
            .where(usage_counters.c.user_id == user_id, usage_counters.c.period_date == day)
            .values(analyses_used=func.greatest(usage_counters.c.analyses_used - 1, 0))
        )
        async with self._engine.begin() as conn:
            await conn.execute(stmt)

    async def used(self, user_id: UUID, day: date) -> int:
        stmt = select(usage_counters.c.analyses_used).where(
            usage_counters.c.user_id == user_id, usage_counters.c.period_date == day
        )
        async with self._engine.connect() as conn:
            return (await conn.execute(stmt)).scalar() or 0
