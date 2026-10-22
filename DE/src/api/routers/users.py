from fastapi import APIRouter

from src.api.deps import ContainerDep, CurrentUser
from src.api.schemas import QuotaOut, UserOut

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me")
async def me(user: CurrentUser) -> UserOut:
    return UserOut.from_user(user)


@router.get("/me/quota")
async def my_quota(user: CurrentUser, c: ContainerDep) -> QuotaOut:
    return QuotaOut.from_status(await c.quota.status(user))
