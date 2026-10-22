from fastapi import APIRouter, Response

from src.api.deps import ContainerDep
from src.api.schemas import AuthResponse, LoginRequest, RefreshRequest, RegisterRequest

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", status_code=201)
async def register(body: RegisterRequest, c: ContainerDep) -> AuthResponse:
    user = await c.auth.register(body.email, body.password, body.name)
    return AuthResponse.build(user, c.tokens.issue_pair(user.id))


@router.post("/login")
async def login(body: LoginRequest, c: ContainerDep) -> AuthResponse:
    user, pair = await c.auth.login(body.email, body.password)
    return AuthResponse.build(user, pair)


@router.post("/refresh")
async def refresh(body: RefreshRequest, c: ContainerDep) -> AuthResponse:
    user, pair = await c.auth.refresh(body.refresh_token)
    return AuthResponse.build(user, pair)


@router.post("/logout", status_code=204)
async def logout(body: RefreshRequest, c: ContainerDep) -> Response:
    await c.auth.logout(body.refresh_token)
    return Response(status_code=204)
