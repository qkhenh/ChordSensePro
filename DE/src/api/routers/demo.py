from fastapi import APIRouter, HTTPException

from src.api.deps import CurrentUser, IngestionDep
from src.api.schemas import DemoTrackOut, JobAccepted
from src.ingestion.domain.errors import DemoNotFound

router = APIRouter(tags=["demo"])


@router.get("/demo-tracks")
async def demo_tracks(_: CurrentUser, c: IngestionDep) -> list[DemoTrackOut]:
    return [DemoTrackOut(**vars(track)) for track in c.demo.tracks()]


@router.post("/analyze/demo/{demo_id}", status_code=202)
async def analyze_demo(demo_id: str, user: CurrentUser, c: IngestionDep) -> JobAccepted:
    try:
        job = await c.submission.submit_demo(user, demo_id)
    except DemoNotFound:
        raise HTTPException(404, "Demo track not found") from None
    return JobAccepted(job_id=job.id)
