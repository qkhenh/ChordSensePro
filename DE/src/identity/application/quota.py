"""Daily analysis quota, driven by a versioned subscription policy (FR-18)."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Literal, Protocol
from uuid import UUID

from src.identity.domain.errors import QuotaExceeded
from src.identity.domain.models import Role, User
from src.identity.domain.ports import UsageRepository

Tier = Literal["free", "pro", "education"]


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class SubscriptionPolicy:
    free_daily_quota: int = 5
    student_daily_cap: int = 5


class PolicyProvider(Protocol):
    async def current(self) -> SubscriptionPolicy: ...


class StaticPolicyProvider:
    """Fixed policy. Replaced by the versioned, Music-PM-managed policy later (FR-28)."""

    def __init__(self, policy: SubscriptionPolicy | None = None) -> None:
        self._policy = policy or SubscriptionPolicy()

    async def current(self) -> SubscriptionPolicy:
        return self._policy


def tier_for(role: Role) -> Tier:
    if role is Role.FREE:
        return "free"
    if role is Role.STUDENT:
        return "education"
    return "pro"


@dataclass(frozen=True)
class QuotaStatus:
    tier: Tier
    limit: int | None  # None = unlimited (fair use)
    used: int
    remaining: int | None
    resets_at: datetime


@dataclass(frozen=True)
class ChargeReceipt:
    user_id: UUID
    day: date | None
    charged: bool  # False for unlimited tiers: nothing to refund


class QuotaService:
    def __init__(
        self,
        usage: UsageRepository,
        policy: PolicyProvider,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        self._usage = usage
        self._policy = policy
        self._clock = clock

    def _today(self) -> date:
        return self._clock().astimezone(UTC).date()

    def _next_reset(self) -> datetime:
        return datetime.combine(self._today() + timedelta(days=1), time.min, UTC)

    async def _limit_for(self, role: Role) -> int | None:
        tier = tier_for(role)
        if tier == "pro":
            return None
        policy = await self._policy.current()
        return policy.free_daily_quota if tier == "free" else policy.student_daily_cap

    async def status(self, user: User) -> QuotaStatus:
        limit = await self._limit_for(user.role)
        used = await self._usage.used(user.id, self._today()) if limit is not None else 0
        return QuotaStatus(
            tier=tier_for(user.role),
            limit=limit,
            used=used,
            remaining=None if limit is None else max(limit - used, 0),
            resets_at=self._next_reset(),
        )

    async def consume(self, user: User) -> ChargeReceipt:
        """Charge one analysis (call at job acceptance). Raises ``QuotaExceeded``."""
        limit = await self._limit_for(user.role)
        if limit is None:
            return ChargeReceipt(user.id, None, charged=False)
        day = self._today()
        if limit <= 0 or await self._usage.try_consume(user.id, day, limit) is None:
            reset_at = self._next_reset()
            retry = math.ceil((reset_at - self._clock()).total_seconds())
            raise QuotaExceeded(limit, reset_at, retry)
        return ChargeReceipt(user.id, day, charged=True)

    async def refund(self, receipt: ChargeReceipt) -> None:
        """Give a charge back to the day it was made (job failed or timed out).

        Not idempotent per receipt: callers guard with ``jobs.quota_charged`` so a job is
        refunded at most once.
        """
        if receipt.charged and receipt.day is not None:
            await self._usage.refund(receipt.user_id, receipt.day)
