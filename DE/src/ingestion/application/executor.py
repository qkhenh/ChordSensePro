"""Runs one analysis job from ``queued`` to ``done`` or ``failed``.

``run`` never raises (except for cancellation): every failure ends in a ``failed`` job, a
single quota refund and a cleaned work directory. Timeout (UC1 EF1) is enforced here so every
runner — in-process or Celery — gets it.
"""

import asyncio
import shutil
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import structlog

from src.analysis_contract.models import (
    AnalysisRequest,
    ErrorCode,
    JobError,
    JobProgress,
    JobStage,
    JobState,
    PlaybackSource,
    SourceKind,
)
from src.analysis_contract.ports import AnalysisEngine, AnalysisError
from src.identity.application.quota import ChargeReceipt, QuotaService
from src.ingestion.application.charts import build_chart
from src.ingestion.application.normalizer import Normalizer
from src.ingestion.domain.errors import IngestionError
from src.ingestion.domain.job import Job, advance, to_progress
from src.ingestion.domain.ports import (
    AnalysisCache,
    ChartRepository,
    JobEvents,
    JobRepository,
    MediaResolverPort,
)
from src.ingestion.settings import IngestSettings

log = structlog.get_logger("ingestion")

_FINAL = (JobState.DONE, JobState.FAILED)
_INTERRUPTED = JobError(
    code=ErrorCode.AI_FAILURE, message="The analysis was interrupted", retryable=True
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _keep_audio(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.move(src, dest)


class JobExecutor:
    def __init__(
        self,
        *,
        jobs: JobRepository,
        charts: ChartRepository,
        cache: AnalysisCache,
        engine: AnalysisEngine,
        normalizer: Normalizer,
        resolver: MediaResolverPort,
        quota: QuotaService,
        events: JobEvents,
        settings: IngestSettings,
        clock: Callable[[], datetime] = _utcnow,
        timeout_errors: tuple[type[BaseException], ...] = (TimeoutError,),
    ) -> None:
        self._jobs = jobs
        self._charts = charts
        self._cache = cache
        self._engine = engine
        self._normalizer = normalizer
        self._resolver = resolver
        self._quota = quota
        self._events = events
        self._settings = settings
        self._clock = clock
        self._timeout_errors = timeout_errors

    def _work_dir(self, job_id: UUID) -> Path:
        return self._settings.data_dir / "tmp" / str(job_id)

    # --- public --------------------------------------------------------------------

    async def run(self, job_id: UUID) -> None:
        job = await self._jobs.get(job_id)
        if job is None or job.state in _FINAL:
            return
        try:
            await self._execute(job)
        except Exception as exc:
            await self._fail(job_id, self._error_for(job_id, exc))
        finally:
            await asyncio.to_thread(shutil.rmtree, self._work_dir(job_id), True)

    async def recover_orphans(self) -> int:
        """Fail jobs left ``queued``/``processing`` by a dead worker (restart, crash)."""
        cutoff = self._clock() - timedelta(seconds=self._settings.job_timeout_s)
        recovered = 0
        for job in await self._jobs.list_unfinished():
            if job.updated_at <= cutoff:
                await self._fail(job.id, _INTERRUPTED)
                await asyncio.to_thread(shutil.rmtree, self._work_dir(job.id), True)
                recovered += 1
        return recovered

    # --- the happy path --------------------------------------------------------------

    async def _execute(self, job: Job) -> None:
        work_dir = _ensure_dir(self._work_dir(job.id))
        playback: PlaybackSource | None = None
        artist: str | None = None
        title = job.title

        if job.source_kind == SourceKind.URL:
            job = await self._move(job, JobStage.FETCHING, 0.05)
            media = await self._resolver.resolve(job.source_url or "", work_dir)
            source, playback, artist, title = media.path, media.playback, media.artist, media.title
            job = await self._move(job, JobStage.NORMALIZING, 0.15)
        else:
            job = await self._move(job, JobStage.NORMALIZING, 0.15)
            source = Path(job.input_path or "")

        audio = await self._normalizer.normalize(
            source, _ensure_dir(work_dir / "norm"), source.stem
        )
        job = replace(job, content_hash=audio.content_hash)
        await self._jobs.save(job)

        cached = await self._cache.get(audio.content_hash)
        if cached is not None:
            result = cached.model_copy(update={"job_id": job.id})
        else:
            result, job = await self._analyze(job, audio.content_hash, audio.path)

        if playback is None:  # the chart plays its own WAV, kept under its content hash
            kept = self._settings.data_dir / "audio" / f"{audio.content_hash}.wav"
            await asyncio.to_thread(_keep_audio, audio.path, kept)
        chart = build_chart(
            job, result, now=self._clock(), playback=playback, artist=artist, title=title
        )
        await self._charts.add(chart)
        if cached is None:
            await self._cache.put(result)
        if job.fingerprint:
            await self._cache.remember_fingerprint(job.fingerprint, audio.content_hash)

        done = await self._advance(job, JobState.DONE, chart_id=chart.id)
        if cached is not None:  # same content as an earlier analysis: no quota spent
            await self._refund(done)

    async def _analyze(self, job: Job, content_hash: str, audio_path: Path):
        latest = job

        async def on_progress(progress: JobProgress) -> None:
            nonlocal latest
            if progress.state != JobState.PROCESSING or progress.stage is None:
                return  # engines report processing steps only; anything else is ignored
            latest = await self._move(latest, progress.stage, progress.progress)

        request = AnalysisRequest(
            job_id=job.id,
            content_hash=content_hash,
            source_kind=job.source_kind,
            audio_path=str(audio_path),
            source_url=job.source_url,
        )
        async with asyncio.timeout(self._settings.job_timeout_s):
            result = await self._engine.analyze(request, on_progress)
        return result, latest

    # --- state changes ----------------------------------------------------------------

    async def _advance(self, job: Job, state: JobState, **changes) -> Job:
        moved = advance(job, state=state, now=self._clock(), **changes)
        await self._jobs.save(moved)
        await self._events.publish(to_progress(moved))
        return moved

    async def _move(self, job: Job, stage: JobStage, progress: float) -> Job:
        return await self._advance(job, JobState.PROCESSING, stage=stage, progress=progress)

    async def _fail(self, job_id: UUID, error: JobError) -> None:
        job = await self._jobs.get(job_id)
        if job is None or job.state in _FINAL:
            return
        failed = await self._advance(job, JobState.FAILED, error=error)
        await self._refund(failed)

    async def _refund(self, job: Job) -> None:
        """Give the charge back at most once, to the day it was taken from."""
        if not job.quota_charged or job.charged_on is None:
            return
        if await self._jobs.try_mark_refunded(job.id):
            await self._quota.refund(ChargeReceipt(job.owner_id, job.charged_on, charged=True))

    def _error_for(self, job_id: UUID, exc: Exception) -> JobError:
        if isinstance(exc, IngestionError | AnalysisError):
            return JobError(code=exc.code, message=exc.message, retryable=exc.retryable)
        if isinstance(exc, self._timeout_errors):
            return JobError(
                code=ErrorCode.ANALYSIS_TIMEOUT,
                message="The analysis took too long",
                retryable=True,
            )
        log.exception("job_crashed", job_id=str(job_id))
        return JobError(
            code=ErrorCode.AI_FAILURE, message="The analysis failed unexpectedly", retryable=True
        )
