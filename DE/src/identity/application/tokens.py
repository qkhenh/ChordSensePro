"""Signed JWT access/refresh tokens (HS256) with revocation.

Tokens carry only identity (``sub``), kind (``type``) and a unique ``jti``; the user's
role and active flag are always read from the database, so role changes and
deactivation take effect immediately.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal
from uuid import UUID, uuid4

import jwt

from src.identity.domain.errors import InvalidToken, TokenExpired, TokenRevoked
from src.identity.domain.ports import RevocationStore

TokenType = Literal["access", "refresh"]
ALGORITHM = "HS256"
MIN_SECRET_LENGTH = 32
_REQUIRED = ["sub", "exp", "iat", "jti", "type"]


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class TokenPair:
    access_token: str
    refresh_token: str
    expires_in: int  # access token lifetime, seconds
    token_type: str = "bearer"  # noqa: S105 - OAuth scheme name, not a secret


@dataclass(frozen=True)
class TokenClaims:
    user_id: UUID
    token_type: str
    jti: str
    expires_at: datetime


class TokenService:
    def __init__(
        self,
        secret: str,
        *,
        access_ttl: timedelta,
        refresh_ttl: timedelta,
        revocations: RevocationStore,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        if len(secret) < MIN_SECRET_LENGTH:
            raise ValueError(f"signing secret must be at least {MIN_SECRET_LENGTH} characters")
        self._secret = secret
        self._access_ttl = access_ttl
        self._refresh_ttl = refresh_ttl
        self._revocations = revocations
        self._clock = clock

    def _issue(self, user_id: UUID, kind: TokenType, ttl: timedelta) -> str:
        now = int(self._clock().timestamp())
        payload = {
            "sub": str(user_id),
            "type": kind,
            "jti": uuid4().hex,
            "iat": now,
            "exp": now + int(ttl.total_seconds()),
        }
        return jwt.encode(payload, self._secret, algorithm=ALGORITHM)

    def issue_pair(self, user_id: UUID) -> TokenPair:
        return TokenPair(
            access_token=self._issue(user_id, "access", self._access_ttl),
            refresh_token=self._issue(user_id, "refresh", self._refresh_ttl),
            expires_in=int(self._access_ttl.total_seconds()),
        )

    async def verify(self, token: str, expected_type: TokenType) -> TokenClaims:
        try:
            payload = jwt.decode(
                token,
                self._secret,
                algorithms=[ALGORITHM],  # pins the algorithm: rejects "none" and HS384/512
                options={"require": _REQUIRED, "verify_exp": False, "verify_iat": False},
            )
            user_id = UUID(payload["sub"])
            expires_at = datetime.fromtimestamp(payload["exp"], UTC)
        except (jwt.InvalidTokenError, ValueError, TypeError, KeyError, OverflowError) as exc:
            raise InvalidToken from exc
        # Expiry is checked against the injected clock (deterministic in tests).
        if expires_at <= self._clock():
            raise TokenExpired
        if payload["type"] != expected_type:
            raise InvalidToken(f"Expected a {expected_type} token")
        if await self._revocations.is_revoked(payload["jti"]):
            raise TokenRevoked
        return TokenClaims(user_id, payload["type"], payload["jti"], expires_at)

    async def revoke(self, claims: TokenClaims) -> None:
        await self._revocations.revoke(claims.jti, claims.expires_at)
