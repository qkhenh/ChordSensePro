"""Cut oversized request bodies off while they stream in, before they are parsed or stored."""

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from src.analysis_contract.models import ErrorCode


class _TooLarge(Exception):
    pass


class BodySizeLimit:
    """ASGI middleware: ``limits`` maps an exact path to the largest body it may carry."""

    def __init__(self, app: ASGIApp, *, limits: dict[str, int]) -> None:
        self.app = app
        self._limits = limits

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        limit = self._limits.get(scope["path"]) if scope["type"] == "http" else None
        if limit is None:
            await self.app(scope, receive, send)
            return

        declared = dict(scope["headers"]).get(b"content-length", b"")
        if declared.isdigit() and int(declared) > limit:
            await self._reject(scope, receive, send)
            return

        received = 0
        started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise _TooLarge
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal started
            started = started or message["type"] == "http.response.start"
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _TooLarge:
            if not started:
                await self._reject(scope, receive, send)

    @staticmethod
    async def _reject(scope: Scope, receive: Receive, send: Send) -> None:
        body = {
            "code": ErrorCode.FILE_TOO_LARGE.value,
            "message": "The upload is larger than the 100 MB limit",
        }
        await JSONResponse(body, status_code=413)(scope, receive, send)
