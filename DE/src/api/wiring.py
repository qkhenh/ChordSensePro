"""Production wiring: PostgreSQL repositories behind the identity services."""

from datetime import timedelta

from sqlalchemy.ext.asyncio import AsyncEngine

from src.api.container import Container
from src.identity.application.auth_service import AuthService
from src.identity.application.passwords import PasswordHasher
from src.identity.application.quota import QuotaService, StaticPolicyProvider
from src.identity.application.tokens import TokenService
from src.identity.infrastructure.memory import InMemoryRevocationStore
from src.identity.infrastructure.postgres import PgUsageRepository, PgUserRepository
from src.identity.settings import AuthSettings


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
