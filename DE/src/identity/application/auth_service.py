"""Registration, login and token refresh."""

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from uuid import uuid4

from src.identity.application.passwords import MAX_PASSWORD_BYTES, PasswordHasher
from src.identity.application.tokens import TokenPair, TokenService
from src.identity.domain.errors import (
    AccountDisabled,
    InvalidCredentials,
    InvalidToken,
    WeakPassword,
)
from src.identity.domain.models import Role, User
from src.identity.domain.ports import UserRepository

MIN_PASSWORD_LENGTH = 8


def _utcnow() -> datetime:
    return datetime.now(UTC)


def normalize_email(email: str) -> str:
    return email.strip().lower()


def validate_password(password: str) -> None:
    if len(password) < MIN_PASSWORD_LENGTH or not password.strip():
        raise WeakPassword(f"Password must have at least {MIN_PASSWORD_LENGTH} characters")
    if len(password.encode()) > MAX_PASSWORD_BYTES:
        raise WeakPassword(f"Password must be at most {MAX_PASSWORD_BYTES} bytes long")


class AuthService:
    def __init__(
        self,
        users: UserRepository,
        hasher: PasswordHasher,
        tokens: TokenService,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        self._users = users
        self._hasher = hasher
        self._tokens = tokens
        self._clock = clock
        self._dummy_digest: str | None = None

    async def register(self, email: str, password: str, name: str | None = None) -> User:
        validate_password(password)
        digest = await asyncio.to_thread(self._hasher.hash, password)
        user = User(
            id=uuid4(),
            email=normalize_email(email),
            name=name,
            password_hash=digest,
            role=Role.FREE,
            is_active=True,
            created_at=self._clock(),
        )
        return await self._users.add(user)  # raises EmailTaken

    async def login(self, email: str, password: str) -> tuple[User, TokenPair]:
        user = await self._users.get_by_email(normalize_email(email))
        # Verify against a dummy digest for unknown emails so response time does not
        # reveal which addresses are registered.
        digest = user.password_hash if user else await self._dummy()
        matches = await asyncio.to_thread(self._hasher.verify, password, digest)
        if user is None or not matches:
            raise InvalidCredentials
        if not user.is_active:
            raise AccountDisabled
        return user, self._tokens.issue_pair(user.id)

    async def refresh(self, refresh_token: str) -> tuple[User, TokenPair]:
        """Rotate: the presented refresh token is revoked and a new pair is issued.

        Note: with a shared revocation store, concurrent use of one token must be made
        atomic (e.g. Redis SET NX); the in-memory store is single-process only.
        """
        claims = await self._tokens.verify(refresh_token, "refresh")
        user = await self._users.get_by_id(claims.user_id)
        if user is None:
            raise InvalidToken
        if not user.is_active:
            raise AccountDisabled
        await self._tokens.revoke(claims)
        return user, self._tokens.issue_pair(user.id)

    async def logout(self, refresh_token: str) -> None:
        """Revoke the refresh token. Idempotent and silent: a bad, expired or already
        revoked token is simply ignored, so signing out never fails or leaks anything."""
        try:
            claims = await self._tokens.verify(refresh_token, "refresh")
        except InvalidToken:
            return
        await self._tokens.revoke(claims)

    async def _dummy(self) -> str:
        if self._dummy_digest is None:
            self._dummy_digest = await asyncio.to_thread(self._hasher.hash, "dummy-password")
        return self._dummy_digest
