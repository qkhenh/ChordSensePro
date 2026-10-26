"""Ports of the ingestion context (adapters live in ``infrastructure``)."""

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol
from uuid import UUID

from src.analysis_contract.models import (
    AnalysisResult,
    Chart,
    JobProgress,
    PlaybackSource,
    SourceKind,
)
from src.ingestion.domain.job import Job


class JobRepository(Protocol):
    async def add(self, job: Job) -> Job: ...

    async def get(self, job_id: UUID) -> Job | None: ...

    async def save(self, job: Job) -> None:
        """Persist a change. A job that is already done/failed is never overwritten."""
        ...

    async def list_unfinished(self) -> list[Job]: ...

    async def try_mark_refunded(self, job_id: UUID) -> bool:
        """Atomically flip ``quota_charged`` true -> false; True for the one caller that did."""
        ...


@dataclass(frozen=True)
class ChartSummary:
    """What a library card needs: everything but the chord events."""

    id: UUID
    title: str
    artist: str | None
    source_kind: SourceKind
    playback_kind: str
    duration_ms: int
    tempo_bpm: float
    key: str
    is_customized: bool
    version: int
    updated_at: datetime


class ChartRepository(Protocol):
    async def add(self, chart: Chart) -> None: ...

    async def list(
        self, owner_id: UUID, *, limit: int, after: tuple[datetime, UUID] | None
    ) -> list[ChartSummary]:
        """The owner's charts, newest first (ties by id, descending), strictly after the cursor."""
        ...

    async def get(self, chart_id: UUID, owner_id: UUID) -> Chart | None:
        """None when the chart does not exist *or* belongs to someone else."""
        ...


class AnalysisCache(Protocol):
    async def get(self, content_hash: str) -> AnalysisResult | None: ...

    async def put(self, result: AnalysisResult) -> None: ...

    async def fingerprint_to_hash(self, fingerprint: str) -> str | None: ...

    async def remember_fingerprint(self, fingerprint: str, content_hash: str) -> None: ...


@dataclass(frozen=True)
class ResolvedMedia:
    path: Path
    title: str
    artist: str | None
    playback: PlaybackSource


class MediaResolverPort(Protocol):
    async def resolve(self, url: str, dest_dir: Path) -> ResolvedMedia:
        """Fetch the audio behind an already-vetted URL into ``dest_dir``."""
        ...


class JobRunnerPort(Protocol):
    async def enqueue(self, job_id: UUID) -> None: ...


class JobEvents(Protocol):
    async def publish(self, progress: JobProgress) -> None: ...

    def subscribe(self, job_id: UUID) -> AbstractAsyncContextManager[AsyncIterator[JobProgress]]:
        """Register now, receive everything published from here on; unregisters on exit."""
        ...
