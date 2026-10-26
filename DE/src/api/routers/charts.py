import base64
import binascii
from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError

from src.analysis_contract.models import Chart
from src.api.deps import CurrentUser, IngestionDep
from src.api.schemas import ChartPage, ChartSummaryOut

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
