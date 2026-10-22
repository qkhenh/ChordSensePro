"""In-memory wiring of the identity services for unit and API tests."""

from datetime import UTC, datetime, timedelta

from src.api.container import Container
from src.identity.application.auth_service import AuthService
from src.identity.application.passwords import PasswordHasher
from src.identity.application.quota import QuotaService, StaticPolicyProvider, SubscriptionPolicy
from src.identity.application.tokens import TokenService
from src.identity.infrastructure.memory import (
    InMemoryRevocationStore,
    InMemoryUsageRepository,
    InMemoryUserRepository,
)

SECRET = "unit-test-secret-" + "x" * 48
PASSWORD = "correct horse battery"


class FakeClock:
    def __init__(self, now: datetime | None = None) -> None:
        self.now = now or datetime(2026, 10, 22, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> None:
        self.now += timedelta(**kwargs)


def make_container(clock: FakeClock | None = None, free_quota: int = 5) -> Container:
    clock = clock or FakeClock()
    users = InMemoryUserRepository()
    tokens = TokenService(
        SECRET,
        access_ttl=timedelta(minutes=15),
        refresh_ttl=timedelta(days=14),
        revocations=InMemoryRevocationStore(),
        clock=clock,
    )
    auth = AuthService(users, PasswordHasher(cost=4), tokens, clock=clock)
    quota = QuotaService(
        InMemoryUsageRepository(),
        StaticPolicyProvider(SubscriptionPolicy(free_daily_quota=free_quota)),
        clock=clock,
    )
    return Container(users=users, auth=auth, tokens=tokens, quota=quota)
