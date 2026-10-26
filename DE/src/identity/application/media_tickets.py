"""Short-lived tickets that let an ``<audio>`` element fetch one chart's audio.

A media element cannot send an ``Authorization`` header, so the API hands the signed-in owner a
URL carrying a ticket: an HS256 JWT bound to one user and one chart, valid for minutes. It is a
different token *type* from access/refresh tokens, so none of them can stand in for another.
"""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import jwt

from src.identity.domain.errors import InvalidToken, TokenExpired

ALGORITHM = "HS256"
TOKEN_TYPE = "media"  # noqa: S105 - a token kind, not a secret
_REQUIRED = ["sub", "chart", "exp", "iat", "jti", "type"]


def _utcnow() -> datetime:
    return datetime.now(UTC)


class MediaTicketService:
    def __init__(
        self, secret: str, *, ttl: timedelta, clock: Callable[[], datetime] = _utcnow
    ) -> None:
        self._secret = secret
        self._ttl = ttl
        self._clock = clock

    def issue(self, user_id: UUID, chart_id: UUID) -> tuple[str, datetime]:
        issued = self._clock()
        expires_at = issued + self._ttl
        payload = {
            "sub": str(user_id),
            "chart": str(chart_id),
            "type": TOKEN_TYPE,
            "jti": uuid4().hex,
            "iat": int(issued.timestamp()),
            "exp": int(expires_at.timestamp()),
        }
        return jwt.encode(payload, self._secret, algorithm=ALGORITHM), expires_at

    def verify(self, token: str, chart_id: UUID) -> UUID:
        """The user the ticket was issued to; raises ``InvalidToken``/``TokenExpired``."""
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
        if payload["type"] != TOKEN_TYPE or payload["chart"] != str(chart_id):
            raise InvalidToken("This ticket is not for this chart")
        if expires_at <= self._clock():
            raise TokenExpired
        return user_id
