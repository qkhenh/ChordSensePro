"""Uniform error bodies: ``{"code": "...", "message": "..."}``."""

from datetime import UTC, datetime

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.identity.domain.errors import (
    AccountDisabled,
    EmailTaken,
    Forbidden,
    IdentityError,
    InvalidCredentials,
    InvalidToken,
    NotAuthenticated,
    QuotaExceeded,
    TokenExpired,
    TokenRevoked,
    WeakPassword,
)

# Most specific class first: lookup walks the exception's MRO.
_STATUS_AND_CODE: dict[type[IdentityError], tuple[int, str]] = {
    EmailTaken: (409, "email_taken"),
    InvalidCredentials: (401, "invalid_credentials"),
    AccountDisabled: (403, "account_disabled"),
    WeakPassword: (422, "weak_password"),
    NotAuthenticated: (401, "not_authenticated"),
    TokenExpired: (401, "token_expired"),
    TokenRevoked: (401, "token_revoked"),
    InvalidToken: (401, "invalid_token"),
    Forbidden: (403, "forbidden"),
    QuotaExceeded: (429, "quota_exceeded"),
}


def iso_z(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _lookup(exc: IdentityError) -> tuple[int, str]:
    for cls in type(exc).__mro__:
        if cls in _STATUS_AND_CODE:
            return _STATUS_AND_CODE[cls]
    return 400, "identity_error"


async def _handle_identity_error(_: Request, exc: IdentityError) -> JSONResponse:
    status, code = _lookup(exc)
    body: dict = {"code": code, "message": exc.message}
    headers: dict[str, str] = {}
    if status == 401:
        headers["WWW-Authenticate"] = "Bearer"
    if isinstance(exc, QuotaExceeded):
        body |= {"limit": exc.limit, "resets_at": iso_z(exc.reset_at)}
        headers["Retry-After"] = str(exc.retry_after_seconds)
    return JSONResponse(body, status_code=status, headers=headers)


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(IdentityError, _handle_identity_error)
