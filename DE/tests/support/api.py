"""A minimal FastAPI app around the identity routers plus probe routes for RBAC tests."""

from typing import Annotated

from fastapi import Depends, FastAPI

from src.api.container import Container
from src.api.deps import ContainerDep, current_user, require_roles
from src.api.errors import install_error_handlers
from src.api.routers import auth, users
from src.identity.domain.models import PRO_ROLES, Role, User

ANY, PRO, ADMIN, CONTENT, CONSUME = (
    "/_t/any",
    "/_t/pro",
    "/_t/admin",
    "/_t/content",
    "/_t/consume",
)


def make_app(container: Container) -> FastAPI:
    app = FastAPI()
    app.state.container = container
    install_error_handlers(app)
    app.include_router(auth.router, prefix="/api/v1")
    app.include_router(users.router, prefix="/api/v1")

    @app.get(ANY)
    async def any_user(user: Annotated[User, Depends(current_user)]):
        return {"role": user.role}

    @app.get(PRO)
    async def pro(user: Annotated[User, Depends(require_roles(*PRO_ROLES))]):
        return {"role": user.role}

    @app.get(ADMIN)
    async def admin(user: Annotated[User, Depends(require_roles(Role.IT_ADMIN))]):
        return {"role": user.role}

    @app.get(CONTENT)
    async def content(user: Annotated[User, Depends(require_roles(Role.MUSIC_PM, Role.EXPERT))]):
        return {"role": user.role}

    @app.post(CONSUME)
    async def consume(user: Annotated[User, Depends(current_user)], c: ContainerDep):
        await c.quota.consume(user)
        return {"ok": True}

    return app
