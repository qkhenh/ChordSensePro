"""In-memory wiring of the ingestion services for unit and API tests."""

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from src.analysis_contract.models import JobProgress, SourceKind
from src.identity.application.quota import QuotaService, StaticPolicyProvider, SubscriptionPolicy
from src.identity.domain.models import Role, User
from src.identity.infrastructure.memory import InMemoryUsageRepository
from src.ingestion.application.executor import JobExecutor
from src.ingestion.application.normalizer import NormalizedAudio
from src.ingestion.domain.job import Job, new_job
from src.ingestion.domain.naming import safe_stem
from src.ingestion.infrastructure.events import InMemoryJobEvents
from src.ingestion.infrastructure.media import FakeMediaResolver
from src.ingestion.infrastructure.memory import (
    InMemoryAnalysisCache,
    InMemoryChartRepository,
    InMemoryJobRepository,
)
from src.ingestion.infrastructure.mock_engine import FIXTURES_DIR, MockAnalysisEngine
from src.ingestion.settings import IngestSettings
from tests.support.identity import FakeClock


def _copy_hashing(src: Path, dest: Path) -> str:
    data = src.read_bytes()
    dest.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _stage_input(settings: IngestSettings, job_id: UUID, name: str, content: bytes) -> str:
    work = settings.data_dir / "tmp" / str(job_id)
    work.mkdir(parents=True)
    staged = work / f"{safe_stem(name)}{Path(name).suffix or '.bin'}"
    staged.write_bytes(content)
    return str(staged)


class FakeNormalizer:
    """Pretends to transcode: copies the bytes and hashes them (no ffmpeg needed)."""

    def __init__(self, duration_ms: int = 9000) -> None:
        self.duration_ms = duration_ms
        self.calls = 0

    async def probe(self, path: Path):  # pragma: no cover - not used by the executor
        raise NotImplementedError

    async def normalize(self, src: Path, dest_dir: Path, stem: str) -> NormalizedAudio:
        self.calls += 1
        dest = dest_dir / f"{safe_stem(stem)}.wav"
        digest = _copy_hashing(src, dest)
        return NormalizedAudio(dest, digest, self.duration_ms)


@dataclass
class RecordingEvents:
    published: list[JobProgress] = field(default_factory=list)

    async def publish(self, progress: JobProgress) -> None:
        self.published.append(progress)

    def subscribe(self, job_id: UUID):  # pragma: no cover - the executor never subscribes
        raise NotImplementedError


def make_user(role: Role = Role.FREE) -> User:
    return User(
        id=uuid4(),
        email=f"{uuid4().hex[:6]}@example.com",
        name="Ann",
        password_hash="x",
        role=role,
        is_active=True,
        created_at=datetime(2026, 10, 26, tzinfo=UTC),
    )


@dataclass
class World:
    settings: IngestSettings
    clock: FakeClock
    jobs: InMemoryJobRepository
    charts: InMemoryChartRepository
    cache: InMemoryAnalysisCache
    events: RecordingEvents
    usage: InMemoryUsageRepository
    quota: QuotaService
    normalizer: FakeNormalizer
    user: User
    executor: JobExecutor

    async def used(self) -> int:
        return await self.usage.used(self.user.id, self.clock().date())

    async def submit(
        self,
        *,
        name: str = "song.mp3",
        content: bytes = b"audio-bytes",
        kind: SourceKind = SourceKind.FILE,
        url: str | None = None,
        fingerprint: str | None = None,
        charge: bool = True,
    ) -> Job:
        """Do what the Submission service does: charge quota, stage the input, add the job."""
        job_id = uuid4()
        receipt = await self.quota.consume(self.user) if charge else None
        input_path = None if url is not None else _stage_input(self.settings, job_id, name, content)
        job = new_job(
            owner_id=self.user.id,
            source_kind=kind,
            title=Path(name).stem,
            now=self.clock(),
            source_url=url,
            input_path=input_path,
            fingerprint=fingerprint,
            quota_charged=bool(receipt and receipt.charged),
            charged_on=receipt.day if receipt else None,
            job_id=job_id,
        )
        return await self.jobs.add(job)


def make_world(
    tmp_path: Path,
    *,
    engine=None,
    resolver=None,
    job_timeout_s: float = 90,
    job_deadline_s: float = 300,
    free_quota: int = 5,
    role: Role = Role.FREE,
    timeout_errors: tuple[type[BaseException], ...] = (TimeoutError,),
) -> World:
    tmp_path.mkdir(parents=True, exist_ok=True)
    clock = FakeClock()
    settings = IngestSettings(
        data_dir=tmp_path, job_timeout_s=job_timeout_s, job_deadline_s=job_deadline_s
    )
    demo = tmp_path / "resolver-demo.wav"
    demo.write_bytes(b"demo-audio")
    jobs, charts, cache = (
        InMemoryJobRepository(),
        InMemoryChartRepository(),
        InMemoryAnalysisCache(),
    )
    events, usage = RecordingEvents(), InMemoryUsageRepository()
    quota = QuotaService(
        usage, StaticPolicyProvider(SubscriptionPolicy(free_daily_quota=free_quota)), clock=clock
    )
    normalizer = FakeNormalizer()
    executor = JobExecutor(
        jobs=jobs,
        charts=charts,
        cache=cache,
        engine=engine or MockAnalysisEngine(),
        normalizer=normalizer,
        resolver=resolver or FakeMediaResolver(demo),
        quota=quota,
        events=events,
        settings=settings,
        clock=clock,
        timeout_errors=timeout_errors,
    )
    return World(
        settings,
        clock,
        jobs,
        charts,
        cache,
        events,
        usage,
        quota,
        normalizer,
        make_user(role),
        executor,
    )


async def public_dns(host: str, port: int) -> list[str]:
    """Every name resolves to a public address, so tests never touch the network."""
    return ["93.184.216.34"]


def build_ingestion(
    data_dir: Path,
    *,
    quota: QuotaService,
    clock: FakeClock,
    normalizer,
    max_upload_bytes: int | None = None,
    engine=None,
):
    """The in-memory ingestion services behind the HTTP layer (real ffmpeg when asked)."""
    from src.api.container import IngestionContainer
    from src.ingestion.application.demo import DemoCatalog
    from src.ingestion.application.submission import SubmissionService
    from src.ingestion.infrastructure.runner import InProcessJobRunner

    data_dir.mkdir(parents=True, exist_ok=True)
    extra = {} if max_upload_bytes is None else {"max_upload_bytes": max_upload_bytes}
    settings = IngestSettings(data_dir=data_dir, **extra)
    jobs, charts, cache = (
        InMemoryJobRepository(),
        InMemoryChartRepository(),
        InMemoryAnalysisCache(),
    )
    events = InMemoryJobEvents()
    executor = JobExecutor(
        jobs=jobs,
        charts=charts,
        cache=cache,
        engine=engine or MockAnalysisEngine(),
        normalizer=normalizer,
        resolver=FakeMediaResolver(None),
        quota=quota,
        events=events,
        settings=settings,
        clock=clock,
    )
    runner = InProcessJobRunner(executor)
    demo = DemoCatalog(data_dir / "demo", FIXTURES_DIR)
    submission = SubmissionService(
        jobs=jobs,
        charts=charts,
        runner=runner,
        quota=quota,
        normalizer=normalizer,
        cache=cache,
        settings=settings,
        demo=demo,
        clock=clock,
        resolve=public_dns,
    )
    return IngestionContainer(
        submission=submission,
        jobs=jobs,
        charts=charts,
        events=events,
        runner=runner,
        executor=executor,
        demo=demo,
        settings=settings,
    )
