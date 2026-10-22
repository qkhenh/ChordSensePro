"""In-memory adapters: unit tests and single-process dev. Not shared across workers."""

from dataclasses import replace
from datetime import date, datetime
from uuid import UUID

from src.identity.domain.errors import EmailTaken
from src.identity.domain.models import Role, User


def _key(email: str) -> str:
    return email.strip().lower()


class InMemoryUserRepository:
    def __init__(self) -> None:
        self._by_id: dict[UUID, User] = {}

    async def add(self, user: User) -> User:
        if any(_key(u.email) == _key(user.email) for u in self._by_id.values()):
            raise EmailTaken
        self._by_id[user.id] = user
        return user

    async def get_by_id(self, user_id: UUID) -> User | None:
        return self._by_id.get(user_id)

    async def get_by_email(self, email: str) -> User | None:
        return next((u for u in self._by_id.values() if _key(u.email) == _key(email)), None)

    async def set_role(self, user_id: UUID, role: Role) -> None:
        self._by_id[user_id] = replace(self._by_id[user_id], role=role)

    async def set_active(self, user_id: UUID, active: bool) -> None:
        self._by_id[user_id] = replace(self._by_id[user_id], is_active=active)


class InMemoryUsageRepository:
    def __init__(self) -> None:
        self._counts: dict[tuple[UUID, date], int] = {}

    async def try_consume(self, user_id: UUID, day: date, limit: int) -> int | None:
        used = self._counts.get((user_id, day), 0)
        if used >= limit:
            return None
        self._counts[(user_id, day)] = used + 1
        return used + 1

    async def refund(self, user_id: UUID, day: date) -> None:
        key = (user_id, day)
        self._counts[key] = max(self._counts.get(key, 0) - 1, 0)

    async def used(self, user_id: UUID, day: date) -> int:
        return self._counts.get((user_id, day), 0)


class InMemoryRevocationStore:
    def __init__(self) -> None:
        self._revoked: dict[str, datetime] = {}

    async def revoke(self, jti: str, expires_at: datetime) -> None:
        self._revoked[jti] = expires_at

    async def is_revoked(self, jti: str) -> bool:
        return jti in self._revoked
