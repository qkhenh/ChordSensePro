"""Request-id propagation and one structured access-log line per request."""

import re
import time
from uuid import uuid4

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from src.api.errors import error_response

REQUEST_ID_HEADER = "X-Request-ID"
# Client-supplied ids are echoed into headers and logs: accept only a short, safe alphabet.
_SAFE_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")

log = structlog.get_logger("api")


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        supplied = request.headers.get(REQUEST_ID_HEADER, "")
        request_id = supplied if _SAFE_ID.fullmatch(supplied) else uuid4().hex
        request.state.request_id = request_id

        structlog.contextvars.bind_contextvars(request_id=request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # Answered here (not by Starlette's outermost handler) so the 500 still carries
            # the request id and gets an access-log line; the traceback goes to the log only.
            log.exception("unhandled_exception", request_id=request_id)
            response = error_response(500, "internal_error", "Internal server error")
        finally:
            structlog.contextvars.unbind_contextvars("request_id")
        response.headers[REQUEST_ID_HEADER] = request_id
        log.info(
            "request",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return response
