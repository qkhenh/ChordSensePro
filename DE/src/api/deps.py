"""FastAPI dependencies: container access, authentication and role checks."""

from typing import Annotated

from fastapi import Depends, Request

from src.api.container import Container
from src.identity.domain.errors import Forbidden, InvalidToken, NotAuthenticated
from src.identity.domain.models import Role, User


def get_container(request: Request) -> Container:
    return request.app.state.container


ContainerDep = Annotated[Container, Depends(get_container)]


def _bearer_token(request: Request) -> str:
    parts = request.headers.get("authorization", "").split(" ")
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1]:
        raise NotAuthenticated
    return parts[1]


async def current_user(request: Request, container: ContainerDep) -> User:
    claims = await container.tokens.verify(_bearer_token(request), "access")
    user = await container.users.get_by_id(claims.user_id)
    if user is None or not user.is_active:
        raise InvalidToken
    return user


CurrentUser = Annotated[User, Depends(current_user)]


def require_roles(*roles: Role):
    allowed = frozenset(roles)

    async def dependency(user: CurrentUser) -> User:
        if user.role not in allowed:
            raise Forbidden
        return user

    return dependency
