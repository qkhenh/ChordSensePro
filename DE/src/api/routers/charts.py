from uuid import UUID

from fastapi import APIRouter, HTTPException

from src.analysis_contract.models import Chart
from src.api.deps import CurrentUser, IngestionDep

router = APIRouter(prefix="/charts", tags=["charts"])


@router.get("/{chart_id}")
async def get_chart(chart_id: UUID, user: CurrentUser, c: IngestionDep) -> Chart:
    chart = await c.charts.get(chart_id, user.id)
    if chart is None:  # also covers someone else's chart
        raise HTTPException(404, "Chart not found")
    return chart
