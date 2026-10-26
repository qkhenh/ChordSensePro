"""PostgreSQL adapters (SQLAlchemy Core, asyncpg) for the ingestion ports."""

from uuid import UUID

from sqlalchemy import null, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine

from src.analysis_contract.models import (
    AnalysisResult,
    Chart,
    JobError,
    JobStage,
    JobState,
    SourceKind,
)
from src.ingestion.domain.job import Job
from src.persistence.tables import analyses, charts, jobs, source_fingerprints

_FINAL = (JobState.DONE.value, JobState.FAILED.value)
_UNFINISHED = (JobState.QUEUED.value, JobState.PROCESSING.value)


def _job_from_row(row) -> Job:
    return Job(
        id=row.id,
        owner_id=row.owner_id,
        source_kind=SourceKind(row.source_kind),
        title=row.title,
        created_at=row.created_at,
        updated_at=row.updated_at,
        source_url=row.source_url,
        input_path=row.input_path,
        fingerprint=row.fingerprint,
        content_hash=row.content_hash,
        state=JobState(row.state),
        stage=JobStage(row.stage) if row.stage else None,
        progress=round(float(row.progress), 4),  # REAL column: drop float32 noise
        error=JobError.model_validate(row.error) if row.error else None,
        result_chart_id=row.result_chart_id,
        quota_charged=row.quota_charged,
        charged_on=row.charged_on,
    )


class PgJobRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def add(self, job: Job) -> Job:
        stmt = insert(jobs).values(
            id=job.id,
            owner_id=job.owner_id,
            source_kind=job.source_kind.value,
            source_url=job.source_url,
            title=job.title,
            input_path=job.input_path,
            fingerprint=job.fingerprint,
            content_hash=job.content_hash,
            state=job.state.value,
            stage=job.stage.value if job.stage else None,
            progress=job.progress,
            error=null() if job.error is None else job.error.model_dump(mode="json"),
            result_chart_id=job.result_chart_id,
            quota_charged=job.quota_charged,
            charged_on=job.charged_on,
            created_at=job.created_at,
            updated_at=job.updated_at,
        )
        async with self._engine.begin() as conn:
            await conn.execute(stmt)
        return job

    async def get(self, job_id: UUID) -> Job | None:
        async with self._engine.connect() as conn:
            row = (await conn.execute(select(jobs).where(jobs.c.id == job_id))).first()
        return _job_from_row(row) if row else None

    async def save(self, job: Job) -> None:
        # quota_charged is deliberately not written: only try_mark_refunded may flip it.
        stmt = (
            update(jobs)
            .where(jobs.c.id == job.id, jobs.c.state.not_in(_FINAL))
            .values(
                state=job.state.value,
                stage=job.stage.value if job.stage else None,
                progress=job.progress,
                error=null() if job.error is None else job.error.model_dump(mode="json"),
                result_chart_id=job.result_chart_id,
                content_hash=job.content_hash,
                input_path=job.input_path,
                updated_at=job.updated_at,
                finished_at=job.updated_at if job.state.value in _FINAL else None,
            )
        )
        async with self._engine.begin() as conn:
            await conn.execute(stmt)

    async def list_unfinished(self) -> list[Job]:
        stmt = select(jobs).where(jobs.c.state.in_(_UNFINISHED)).order_by(jobs.c.created_at)
        async with self._engine.connect() as conn:
            return [_job_from_row(row) for row in await conn.execute(stmt)]

    async def try_mark_refunded(self, job_id: UUID) -> bool:
        stmt = (
            update(jobs)
            .where(jobs.c.id == job_id, jobs.c.quota_charged)
            .values(quota_charged=False)
            .returning(jobs.c.id)
        )
        async with self._engine.begin() as conn:
            return (await conn.execute(stmt)).first() is not None


class PgChartRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def add(self, chart: Chart) -> None:
        stmt = insert(charts).values(
            id=chart.id,
            owner_id=chart.owner_id,
            title=chart.title,
            artist=chart.artist,
            source=chart.source.model_dump(mode="json"),
            metadata=chart.metadata.model_dump(mode="json"),
            ai_events=[e.model_dump(mode="json") for e in chart.ai_events],
            overrides=[o.model_dump(mode="json") for o in chart.overrides],
            is_customized=chart.is_customized,
            forked_from=chart.forked_from,
            content_hash=chart.source.content_hash,
            version=chart.version,
            created_at=chart.created_at,
            updated_at=chart.updated_at,
        )
        async with self._engine.begin() as conn:
            await conn.execute(stmt)

    async def get(self, chart_id: UUID, owner_id: UUID) -> Chart | None:
        stmt = select(charts).where(charts.c.id == chart_id, charts.c.owner_id == owner_id)
        async with self._engine.connect() as conn:
            row = (await conn.execute(stmt)).first()
        if row is None:
            return None
        data = row._mapping
        return Chart.model_validate(
            {
                "id": data["id"],
                "owner_id": data["owner_id"],
                "title": data["title"],
                "artist": data["artist"],
                "source": data["source"],
                "metadata": data["metadata"],
                "ai_events": data["ai_events"],
                "overrides": data["overrides"],
                "is_customized": data["is_customized"],
                "forked_from": data["forked_from"],
                "version": data["version"],
                "created_at": data["created_at"],
                "updated_at": data["updated_at"],
            }
        )


class PgAnalysisCache:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def get(self, content_hash: str) -> AnalysisResult | None:
        stmt = select(analyses.c.result).where(analyses.c.content_hash == content_hash)
        async with self._engine.connect() as conn:
            value = (await conn.execute(stmt)).scalar()
        return AnalysisResult.model_validate(value) if value else None

    async def put(self, result: AnalysisResult) -> None:
        stmt = (
            insert(analyses)
            .values(
                content_hash=result.content_hash,
                result=result.model_dump(mode="json"),
                model_version=result.metadata.model_version,
                pipeline_version=result.metadata.pipeline_version,
            )
            .on_conflict_do_nothing(index_elements=[analyses.c.content_hash])
        )
        async with self._engine.begin() as conn:
            await conn.execute(stmt)

    async def fingerprint_to_hash(self, fingerprint: str) -> str | None:
        stmt = select(source_fingerprints.c.content_hash).where(
            source_fingerprints.c.fingerprint == fingerprint
        )
        async with self._engine.connect() as conn:
            return (await conn.execute(stmt)).scalar()

    async def remember_fingerprint(self, fingerprint: str, content_hash: str) -> None:
        stmt = (
            insert(source_fingerprints)
            .values(fingerprint=fingerprint, content_hash=content_hash)
            .on_conflict_do_update(
                index_elements=[source_fingerprints.c.fingerprint],
                set_={"content_hash": content_hash},
            )
        )
        async with self._engine.begin() as conn:
            await conn.execute(stmt)
