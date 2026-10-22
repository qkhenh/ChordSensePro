"""Wired services shared by the API layer (kept on ``app.state.container``)."""

from dataclasses import dataclass

from src.identity.application.auth_service import AuthService
from src.identity.application.quota import QuotaService
from src.identity.application.tokens import TokenService
from src.identity.domain.ports import UserRepository


@dataclass(frozen=True)
class Container:
    users: UserRepository
    auth: AuthService
    tokens: TokenService
    quota: QuotaService
