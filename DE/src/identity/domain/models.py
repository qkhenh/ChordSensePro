"""Identity domain: roles and the user entity."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from uuid import UUID


class Role(StrEnum):
    """The eight roles of NFR-10. ``GUEST`` is anonymous and never stored."""

    GUEST = "guest"
    FREE = "free"
    PAID = "paid"
    TEACHER = "teacher"
    STUDENT = "student"
    EXPERT = "expert"
    MUSIC_PM = "music_pm"
    IT_ADMIN = "it_admin"


PERSISTED_ROLES = frozenset(r for r in Role if r is not Role.GUEST)

# Roles that unlock the practice features of the Pro plan (Education seats included).
PRO_ROLES = frozenset({Role.PAID, Role.TEACHER, Role.STUDENT})


@dataclass(frozen=True)
class User:
    id: UUID
    email: str
    name: str | None
    password_hash: str
    role: Role
    is_active: bool
    created_at: datetime
