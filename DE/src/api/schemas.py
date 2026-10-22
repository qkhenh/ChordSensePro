"""Request/response models of the identity endpoints."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from src.identity.application.quota import QuotaStatus, Tier
from src.identity.application.tokens import TokenPair
from src.identity.domain.models import Role, User


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RegisterRequest(_Strict):
    email: EmailStr
    password: str
    name: str | None = Field(default=None, max_length=200)


class LoginRequest(_Strict):
    email: str
    password: str


class RefreshRequest(_Strict):
    refresh_token: str


class UserOut(_Strict):
    id: UUID
    email: str
    name: str | None
    role: Role
    created_at: datetime

    @classmethod
    def from_user(cls, user: User) -> "UserOut":
        return cls(
            id=user.id,
            email=user.email,
            name=user.name,
            role=user.role,
            created_at=user.created_at,
        )


class AuthResponse(_Strict):
    user: UserOut
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int

    @classmethod
    def build(cls, user: User, pair: TokenPair) -> "AuthResponse":
        return cls(
            user=UserOut.from_user(user),
            access_token=pair.access_token,
            refresh_token=pair.refresh_token,
            token_type=pair.token_type,
            expires_in=pair.expires_in,
        )


class QuotaOut(_Strict):
    tier: Tier
    limit: int | None
    used: int
    remaining: int | None
    resets_at: datetime

    @classmethod
    def from_status(cls, status: QuotaStatus) -> "QuotaOut":
        return cls(
            tier=status.tier,
            limit=status.limit,
            used=status.used,
            remaining=status.remaining,
            resets_at=status.resets_at,
        )
