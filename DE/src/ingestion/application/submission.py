"""Accepting an analysis request: cheap synchronous checks, quota, job row, queue.

Everything slow (decoding, hashing the waveform, the analysis itself) happens later in the
``JobExecutor``; this service only does what fits inside the 300 ms budget of NFR-06.
"""

import asyncio
import hashlib
import re
import shutil
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlsplit
from uuid import UUID, uuid4

import structlog

from src.analysis_contract.models import ErrorCode, JobError, JobState, SourceKind
from src.identity.application.quota import ChargeReceipt, QuotaService
from src.identity.domain.models import User
from src.ingestion.application.charts import build_chart
from src.ingestion.application.normalizer import Normalizer
from src.ingestion.application.ssrf import Resolver, check_url, system_resolver
from src.ingestion.domain.errors import IngestionError
from src.ingestion.domain.job import Job, advance, new_job
from src.ingestion.domain.naming import safe_stem, safe_title
from src.ingestion.domain.ports import AnalysisCache, ChartRepository, JobRepository, JobRunnerPort
from src.ingestion.domain.upload_rules import check_format, check_limits
from src.ingestion.settings import IngestSettings

log = structlog.get_logger("ingestion")

_SUFFIX = re.compile(r"\.[a-z0-9]{1,5}")
_HEAD_BYTES = 16


def _utcnow() -> datetime:
    return datetime.now(UTC)


class _Sink:
    """Synchronous file writer, so async code never touches ``open`` directly."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("wb")

    def write(self, data: bytes) -> None:
        self._handle.write(data)

    def close(self) -> None:
        self._handle.close()


def _read_head(path: Path) -> bytes:
    with path.open("rb") as handle:
        return handle.read(_HEAD_BYTES)


def _suffix_of(filename: str) -> str:
    suffix = Path(filename.replace("\\", "/")).suffix.lower()
    return suffix if _SUFFIX.fullmatch(suffix) else ".bin"


def _title_for_url(url: str) -> str:
    parts = urlsplit(url)
    last = unquote(parts.path.rstrip("/").rsplit("/", 1)[-1])
    title = safe_title(last)
    return parts.hostname or title if title == "Untitled" else title


class SubmissionService:
    def __init__(
        self,
        *,
        jobs: JobRepository,
        charts: ChartRepository,
        runner: JobRunnerPort,
        quota: QuotaService,
        normalizer: Normalizer,
        cache: AnalysisCache,
        settings: IngestSettings,
        clock: Callable[[], datetime] = _utcnow,
        resolve: Resolver = system_resolver,
    ) -> None:
        self._jobs = jobs
        self._charts = charts
        self._runner = runner
        self._quota = quota
        self._normalizer = normalizer
        self._cache = cache
        self._settings = settings
        self._clock = clock
        self._resolve = resolve

    def _work_dir(self, job_id: UUID) -> Path:
        return self._settings.data_dir / "tmp" / str(job_id)

    async def _discard(self, job_id: UUID) -> None:
        await asyncio.to_thread(shutil.rmtree, self._work_dir(job_id), True)

    # --- file upload -----------------------------------------------------------------

    async def submit_file(self, user: User, filename: str, chunks: AsyncIterator[bytes]) -> Job:
        job_id = uuid4()
        try:
            path, size, fingerprint = await self._receive(job_id, filename, chunks)
            head = await asyncio.to_thread(_read_head, path)
            info = await self._normalizer.probe(path)
            check_format(head, info)
            check_limits(size, info.duration_ms, settings=self._settings)
            cached = await self._from_cache(user, job_id, filename, fingerprint)
            if cached is not None:
                await self._discard(job_id)
                return cached
            job = await self._accept(
                user,
                job_id,
                source_kind=SourceKind.FILE,
                title=safe_title(filename),
                input_path=str(path),
                fingerprint=fingerprint,
            )
        except BaseException:
            await self._discard(job_id)
            raise
        return await self._start(job)

    async def _receive(
        self, job_id: UUID, filename: str, chunks: AsyncIterator[bytes]
    ) -> tuple[Path, int, str]:
        """Stream the upload to disk under a generated name, counting bytes as they arrive."""
        path = self._work_dir(job_id) / f"{safe_stem(filename)}{_suffix_of(filename)}"
        sink, digest, size = _Sink(path), hashlib.sha256(), 0
        try:
            async for chunk in chunks:
                size += len(chunk)
                if size > self._settings.max_upload_bytes:
                    raise IngestionError(
                        ErrorCode.FILE_TOO_LARGE, "the file is larger than the 100 MB limit"
                    )
                sink.write(chunk)
                digest.update(chunk)
        finally:
            sink.close()
        return path, size, digest.hexdigest()

    async def _from_cache(
        self, user: User, job_id: UUID, filename: str, fingerprint: str
    ) -> Job | None:
        """The same bytes were analysed before: hand over a finished job, charge nothing.

        Only when the canonical WAV is still kept, so the new chart can really be played.
        """
        content_hash = await self._cache.fingerprint_to_hash(fingerprint)
        result = await self._cache.get(content_hash) if content_hash else None
        if result is None:
            return None
        kept = self._settings.data_dir / "audio" / f"{result.content_hash}.wav"
        if not await asyncio.to_thread(kept.exists):
            return None
        now = self._clock()
        queued = new_job(
            owner_id=user.id,
            job_id=job_id,
            now=now,
            source_kind=SourceKind.FILE,
            title=safe_title(filename),
            fingerprint=fingerprint,
            quota_charged=False,
        )
        chart = build_chart(queued, result.model_copy(update={"job_id": job_id}), now=now)
        await self._charts.add(chart)
        done = advance(queued, state=JobState.DONE, chart_id=chart.id, now=now)
        return await self._jobs.add(replace(done, content_hash=result.content_hash))

    # --- link ------------------------------------------------------------------------

    async def submit_url(self, user: User, url: str) -> Job:
        await check_url(url, self._resolve)  # refuses before any quota is spent
        job_id = uuid4()
        try:
            job = await self._accept(
                user,
                job_id,
                source_kind=SourceKind.URL,
                title=_title_for_url(url),
                source_url=url,
            )
        except BaseException:
            await self._discard(job_id)
            raise
        return await self._start(job)

    # --- shared ----------------------------------------------------------------------

    async def _accept(self, user: User, job_id: UUID, **fields) -> Job:
        """Charge the quota and persist the queued job (the charge is undone if that fails)."""
        receipt = await self._quota.consume(user)
        job = new_job(
            owner_id=user.id,
            job_id=job_id,
            now=self._clock(),
            quota_charged=receipt.charged,
            charged_on=receipt.day,
            **fields,
        )
        try:
            return await self._jobs.add(job)
        except BaseException:
            await self._quota.refund(receipt)
            raise

    async def _start(self, job: Job) -> Job:
        try:
            await self._runner.enqueue(job.id)
        except Exception:
            log.exception("enqueue_failed", job_id=str(job.id))
            error = JobError(
                code=ErrorCode.AI_FAILURE, message="Could not start the analysis", retryable=True
            )
            await self._jobs.save(
                advance(job, state=JobState.FAILED, error=error, now=self._clock())
            )
            if (
                job.quota_charged
                and job.charged_on is not None
                and await self._jobs.try_mark_refunded(job.id)
            ):
                await self._quota.refund(ChargeReceipt(job.owner_id, job.charged_on, charged=True))
            await self._discard(job.id)
            raise IngestionError(error.code, error.message, retryable=True) from None
        return job
