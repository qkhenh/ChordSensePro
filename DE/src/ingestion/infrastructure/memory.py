"""In-memory adapters for unit tests and the single-process dev server."""

from dataclasses import replace
from datetime import datetime
from uuid import UUID

from src.analysis_contract.models import AnalysisResult, Chart, JobState
from src.ingestion.domain.job import Job
from src.ingestion.domain.ports import ChartSummary

_FINAL = (JobState.DONE, JobState.FAILED)


def _summary(chart: Chart) -> ChartSummary:
    meta = chart.metadata
    return ChartSummary(
        id=chart.id,
        title=chart.title,
        artist=chart.artist,
        source_kind=chart.source.kind,
        playback_kind=chart.source.playback.kind,
        duration_ms=meta.duration_ms,
        tempo_bpm=meta.beat_grid.tempo_bpm,
        key=f"{meta.key.tonic.value} {meta.key.mode}",
        is_customized=chart.is_customized,
        version=chart.version,
        updated_at=chart.updated_at,
    )


class InMemoryJobRepository:
    def __init__(self) -> None:
        self._jobs: dict[UUID, Job] = {}

    async def add(self, job: Job) -> Job:
        self._jobs[job.id] = job
        return job

    async def get(self, job_id: UUID) -> Job | None:
        return self._jobs.get(job_id)

    async def save(self, job: Job) -> None:
        current = self._jobs.get(job.id)
        if current is not None and current.state in _FINAL:
            return
        self._jobs[job.id] = job

    async def list_unfinished(self) -> list[Job]:
        return [j for j in self._jobs.values() if j.state not in _FINAL]

    async def try_mark_refunded(self, job_id: UUID) -> bool:
        job = self._jobs.get(job_id)
        if job is None or not job.quota_charged:
            return False
        self._jobs[job_id] = replace(job, quota_charged=False)
        return True


class InMemoryChartRepository:
    def __init__(self) -> None:
        self._charts: dict[UUID, Chart] = {}

    async def add(self, chart: Chart) -> None:
        self._charts[chart.id] = chart

    async def get(self, chart_id: UUID, owner_id: UUID) -> Chart | None:
        chart = self._charts.get(chart_id)
        return chart if chart is not None and chart.owner_id == owner_id else None

    async def list(
        self, owner_id: UUID, *, limit: int, after: tuple[datetime, UUID] | None
    ) -> list[ChartSummary]:
        mine = sorted(
            (c for c in self._charts.values() if c.owner_id == owner_id),
            key=lambda c: (c.updated_at, c.id),
            reverse=True,
        )
        if after is not None:
            mine = [c for c in mine if (c.updated_at, c.id) < after]
        return [_summary(c) for c in mine[:limit]]


class InMemoryAnalysisCache:
    def __init__(self) -> None:
        self._results: dict[str, AnalysisResult] = {}
        self._fingerprints: dict[str, str] = {}

    async def get(self, content_hash: str) -> AnalysisResult | None:
        return self._results.get(content_hash)

    async def put(self, result: AnalysisResult) -> None:
        self._results[result.content_hash] = result

    async def fingerprint_to_hash(self, fingerprint: str) -> str | None:
        return self._fingerprints.get(fingerprint)

    async def remember_fingerprint(self, fingerprint: str, content_hash: str) -> None:
        self._fingerprints[fingerprint] = content_hash
