import asyncio
import base64
import binascii
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse

from src.analysis_contract.models import Chart
from src.api.deps import ContainerDep, CurrentUser, IngestionDep, MediaUser
from src.api.ranges import UNSATISFIABLE, parse_range
from src.api.schemas import AudioTicketOut, ChartPage, ChartSummaryOut

router = APIRouter(prefix="/charts", tags=["charts"])

MAX_PAGE = 100


def _encode_cursor(updated_at: datetime, chart_id: UUID) -> str:
    raw = f"{updated_at.isoformat()}|{chart_id}".encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_cursor(cursor: str) -> tuple[datetime, UUID]:
    try:
        raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)).decode()
        stamp, chart_id = raw.split("|")
        return datetime.fromisoformat(stamp), UUID(chart_id)
    except (binascii.Error, ValueError, UnicodeDecodeError):
        raise RequestValidationError(
            [
                {
                    "loc": ("query", "cursor"),
                    "msg": "not a cursor issued by this API",
                    "type": "value_error",
                }
            ]
        ) from None


def _etag(chart: Chart) -> str:
    return f'W/"{chart.version}-{int(chart.updated_at.timestamp() * 1000)}"'


def _matches(header: str | None, etag: str) -> bool:
    if not header:
        return False
    wanted = [part.strip() for part in header.split(",")]
    return "*" in wanted or etag in wanted


@router.get("")
async def list_charts(
    user: CurrentUser,
    c: IngestionDep,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE)] = 20,
    cursor: str | None = None,
) -> ChartPage:
    after = _decode_cursor(cursor) if cursor is not None else None
    found = await c.charts.list(user.id, limit=limit + 1, after=after)
    page = found[:limit]
    more = len(found) > limit
    return ChartPage(
        items=[ChartSummaryOut(**vars(s)) for s in page],
        next_cursor=_encode_cursor(page[-1].updated_at, page[-1].id) if more else None,
    )


@router.get("/{chart_id}", response_model=Chart)
async def get_chart(
    chart_id: UUID, request: Request, response: Response, user: CurrentUser, c: IngestionDep
) -> Chart | Response:
    chart = await c.charts.get(chart_id, user.id)
    if chart is None:  # also covers someone else's chart
        raise HTTPException(404, "Chart not found")
    headers = {"ETag": _etag(chart), "Cache-Control": "private, no-cache"}
    if _matches(request.headers.get("if-none-match"), headers["ETag"]):
        return Response(status_code=304, headers=headers)
    response.headers.update(headers)
    return chart


# --- audio ---------------------------------------------------------------------------

_CHUNK = 64 * 1024


def _size_of(path: Path) -> int | None:
    return path.stat().st_size if path.is_file() else None


def _read(path: Path, offset: int, length: int) -> bytes:
    with path.open("rb") as handle:
        handle.seek(offset)
        return handle.read(length)


async def _stored_audio(c, chart_id: UUID, user) -> tuple[Path, int]:
    """The chart's WAV and its size; 404 for anything that has no audio of ours to serve."""
    chart = await c.charts.get(chart_id, user.id)
    if chart is None or chart.source.playback.kind != "file":
        raise HTTPException(404, "This chart has no stored audio")
    path = c.settings.data_dir / "audio" / f"{chart.source.content_hash}.wav"
    size = await asyncio.to_thread(_size_of, path)
    if size is None:
        raise HTTPException(404, "This chart has no stored audio")
    return path, size


async def _chunks(path: Path, first: int, last: int) -> AsyncIterator[bytes]:
    offset = first
    while offset <= last:
        data = await asyncio.to_thread(_read, path, offset, min(_CHUNK, last - offset + 1))
        if not data:
            return
        offset += len(data)
        yield data


@router.post("/{chart_id}/audio-ticket")
async def audio_ticket(
    chart_id: UUID, user: CurrentUser, c: IngestionDep, auth: ContainerDep
) -> AudioTicketOut:
    await _stored_audio(c, chart_id, user)
    token, expires_at = auth.tickets.issue(user.id, chart_id)
    return AudioTicketOut(
        url=f"/api/v1/charts/{chart_id}/audio?ticket={token}", expires_at=expires_at
    )


@router.api_route("/{chart_id}/audio", methods=["GET", "HEAD"])
async def chart_audio(
    chart_id: UUID, request: Request, user: MediaUser, c: IngestionDep
) -> Response:
    path, size = await _stored_audio(c, chart_id, user)
    span = parse_range(request.headers.get("range"), size)
    if span == UNSATISFIABLE:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})

    first, last = span if span is not None else (0, size - 1)
    headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "private, max-age=3600",
        "Content-Length": str(last - first + 1),
    }
    status = 200
    if span is not None:
        status = 206
        headers["Content-Range"] = f"bytes {first}-{last}/{size}"
    if request.method == "HEAD":
        return Response(status_code=status, headers=headers, media_type="audio/wav")
    return StreamingResponse(
        _chunks(path, first, last), status_code=status, headers=headers, media_type="audio/wav"
    )
