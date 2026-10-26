"""Production wiring: PostgreSQL repositories behind the identity services."""

from datetime import timedelta

from sqlalchemy.ext.asyncio import AsyncEngine

from src.api.container import Container, IngestionContainer
from src.identity.application.auth_service import AuthService
from src.identity.application.passwords import PasswordHasher
from src.identity.application.quota import QuotaService, StaticPolicyProvider
from src.identity.application.tokens import TokenService
from src.identity.infrastructure.memory import InMemoryRevocationStore
from src.identity.infrastructure.postgres import PgUsageRepository, PgUserRepository
from src.identity.settings import AuthSettings
from src.ingestion.application.demo import DemoCatalog
from src.ingestion.application.executor import JobExecutor
from src.ingestion.application.submission import SubmissionService
from src.ingestion.infrastructure.events import InMemoryJobEvents
from src.ingestion.infrastructure.ffmpeg import FfmpegNormalizer
from src.ingestion.infrastructure.media import FakeMediaResolver
from src.ingestion.infrastructure.mock_engine import FIXTURES_DIR, MockAnalysisEngine
from src.ingestion.infrastructure.postgres import (
    PgAnalysisCache,
    PgChartRepository,
    PgJobRepository,
)
from src.ingestion.infrastructure.runner import InProcessJobRunner
from src.ingestion.settings import IngestSettings


def build_container(settings: AuthSettings, engine: AsyncEngine) -> Container:
    users = PgUserRepository(engine)
    tokens = TokenService(
        settings.jwt_secret.get_secret_value(),
        access_ttl=timedelta(minutes=settings.access_ttl_minutes),
        refresh_ttl=timedelta(days=settings.refresh_ttl_days),
        # Dev only: revoked refresh tokens become valid again after a restart (context.md,
        # open risks). Replace with Redis or a PG table before running for real (M2.6).
        revocations=InMemoryRevocationStore(),
    )
    return Container(
        users=users,
        auth=AuthService(users, PasswordHasher(cost=settings.bcrypt_cost), tokens),
        tokens=tokens,
        quota=QuotaService(PgUsageRepository(engine), StaticPolicyProvider()),
    )


def build_ingestion(
    settings: IngestSettings, engine: AsyncEngine, container: Container
) -> IngestionContainer:
    jobs, charts, cache = (
        PgJobRepository(engine),
        PgChartRepository(engine),
        PgAnalysisCache(engine),
    )
    normalizer, events = FfmpegNormalizer(), InMemoryJobEvents()
    executor = JobExecutor(
        jobs=jobs,
        charts=charts,
        cache=cache,
        engine=MockAnalysisEngine(stage_delay_ms=settings.mock_stage_delay_ms),
        normalizer=normalizer,
        resolver=FakeMediaResolver(None),
        quota=container.quota,
        events=events,
        settings=settings,
    )
    runner = InProcessJobRunner(executor)
    demo = DemoCatalog(settings.data_dir / "demo", FIXTURES_DIR)
    submission = SubmissionService(
        jobs=jobs,
        charts=charts,
        runner=runner,
        quota=container.quota,
        normalizer=normalizer,
        cache=cache,
        settings=settings,
        demo=demo,
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
