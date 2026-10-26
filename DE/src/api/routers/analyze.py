from collections.abc import AsyncIterator
from uuid import UUID

from fastapi import APIRouter, HTTPException, UploadFile

from src.analysis_contract.models import JobProgress
from src.api.deps import CurrentUser, IngestionDep
from src.api.schemas import AnalyzeUrlRequest, JobAccepted
from src.ingestion.domain.job import to_progress

router = APIRouter(prefix="/analyze", tags=["analyze"])

_CHUNK = 64 * 1024


async def _chunks(file: UploadFile) -> AsyncIterator[bytes]:
    while data := await file.read(_CHUNK):
        yield data


@router.post("/file", status_code=202)
async def analyze_file(user: CurrentUser, c: IngestionDep, file: UploadFile) -> JobAccepted:
    job = await c.submission.submit_file(user, file.filename or "upload", _chunks(file))
    return JobAccepted(job_id=job.id)


@router.post("/url", status_code=202)
async def analyze_url(body: AnalyzeUrlRequest, user: CurrentUser, c: IngestionDep) -> JobAccepted:
    job = await c.submission.submit_url(user, body.url)
    return JobAccepted(job_id=job.id)


@router.get("/status/{job_id}")
async def job_status(job_id: UUID, user: CurrentUser, c: IngestionDep) -> JobProgress:
    job = await c.jobs.get(job_id)
    if job is None or job.owner_id != user.id:  # 404, not 403: do not reveal other users' jobs
        raise HTTPException(404, "Job not found")
    return to_progress(job)
