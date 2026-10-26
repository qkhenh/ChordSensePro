"""Wired services shared by the API layer (kept on ``app.state.container``)."""

from dataclasses import dataclass

from src.identity.application.auth_service import AuthService
from src.identity.application.media_tickets import MediaTicketService
from src.identity.application.quota import QuotaService
from src.identity.application.tokens import TokenService
from src.identity.domain.ports import UserRepository
from src.ingestion.application.demo import DemoCatalog
from src.ingestion.application.executor import JobExecutor
from src.ingestion.application.submission import SubmissionService
from src.ingestion.domain.ports import ChartRepository, JobEvents, JobRepository, JobRunnerPort
from src.ingestion.settings import IngestSettings


@dataclass(frozen=True)
class Container:
    users: UserRepository
    auth: AuthService
    tokens: TokenService
    quota: QuotaService
    tickets: MediaTicketService


@dataclass(frozen=True)
class IngestionContainer:
    submission: SubmissionService
    jobs: JobRepository
    charts: ChartRepository
    events: JobEvents
    runner: JobRunnerPort
    executor: JobExecutor
    demo: DemoCatalog
    settings: IngestSettings
