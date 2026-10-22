"""Identity errors. The API layer maps each to an HTTP status and a stable ``code``."""

from datetime import datetime


class IdentityError(Exception):
    default_message = "Identity error"

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.default_message)
        self.message = message or self.default_message


class EmailTaken(IdentityError):
    default_message = "An account with this email already exists"


class InvalidCredentials(IdentityError):
    default_message = "Invalid email or password"


class AccountDisabled(IdentityError):
    default_message = "This account is disabled"


class WeakPassword(IdentityError):
    default_message = "Password is too weak"


class NotAuthenticated(IdentityError):
    default_message = "Authentication required"


class InvalidToken(IdentityError):
    default_message = "Invalid token"


class TokenExpired(InvalidToken):
    default_message = "Token has expired"


class TokenRevoked(InvalidToken):
    default_message = "Token has been revoked"


class Forbidden(IdentityError):
    default_message = "You do not have permission to do this"


class QuotaExceeded(IdentityError):
    default_message = "Daily analysis quota exhausted"

    def __init__(self, limit: int, reset_at: datetime, retry_after_seconds: int) -> None:
        super().__init__()
        self.limit = limit
        self.reset_at = reset_at
        self.retry_after_seconds = retry_after_seconds
