"""Persistence ports of the identity context."""

from datetime import date, datetime
from typing import Protocol
from uuid import UUID

from .models import Role, User


class UserRepository(Protocol):
    async def add(self, user: User) -> User:
        """Insert; raise ``EmailTaken`` if the email exists (case-insensitive)."""
        ...

    async def get_by_id(self, user_id: UUID) -> User | None: ...

    async def get_by_email(self, email: str) -> User | None: ...

    async def set_role(self, user_id: UUID, role: Role) -> None: ...

    async def set_active(self, user_id: UUID, active: bool) -> None: ...


class UsageRepository(Protocol):
    """Per-user, per-UTC-day analysis counters."""

    async def try_consume(self, user_id: UUID, day: date, limit: int) -> int | None:
        """Atomically add one if ``used < limit``; return the new count or ``None``."""
        ...

    async def refund(self, user_id: UUID, day: date) -> None:
        """Subtract one, never going below zero."""
        ...

    async def used(self, user_id: UUID, day: date) -> int: ...


class RevocationStore(Protocol):
    async def revoke(self, jti: str, expires_at: datetime) -> None: ...

    async def is_revoked(self, jti: str) -> bool: ...
