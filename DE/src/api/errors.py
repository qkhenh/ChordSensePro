"""Uniform error bodies: ``{"code": "...", "message": "..."}``."""

from datetime import UTC, datetime

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.analysis_contract.models import ERROR_HTTP_STATUS
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
from src.ingestion.domain.errors import IngestionError

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


_HTTP_CODES = {
    400: "bad_request",
    404: "not_found",
    405: "method_not_allowed",
    413: "payload_too_large",
    415: "unsupported_media_type",
}


def error_response(
    status: int, code: str, message: str, headers: dict[str, str] | None = None, **extra
) -> JSONResponse:
    return JSONResponse({"code": code, "message": message, **extra}, status, headers)


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


# Request parts FastAPI prefixes to every validation location.
_LOCATION_PREFIXES = {"body", "query", "path", "header", "cookie"}


def _field_of(loc: tuple) -> str:
    parts = [str(p) for p in loc if p not in _LOCATION_PREFIXES]
    return ".".join(parts) or "body"


async def _handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    detail = [{"field": _field_of(e["loc"]), "message": e["msg"]} for e in exc.errors()]
    return error_response(422, "validation_error", "Request validation failed", detail=detail)


async def _handle_http_exception(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    code = _HTTP_CODES.get(exc.status_code, f"http_{exc.status_code}")
    return error_response(exc.status_code, code, str(exc.detail), dict(exc.headers or {}))


async def _handle_ingestion_error(_: Request, exc: IngestionError) -> JSONResponse:
    return error_response(ERROR_HTTP_STATUS[exc.code], exc.code.value, exc.message)


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(IdentityError, _handle_identity_error)
    app.add_exception_handler(IngestionError, _handle_ingestion_error)
    app.add_exception_handler(RequestValidationError, _handle_validation_error)
    app.add_exception_handler(StarletteHTTPException, _handle_http_exception)
