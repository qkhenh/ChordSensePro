"""Live job progress over WebSocket (NFR-06). Polling ``GET /analyze/status`` stays the fallback.

Browsers cannot set headers on a WebSocket, so the access token travels as a subprotocol:
``new WebSocket(url, ['bearer', accessToken])``. The handshake is always accepted and a
refusal is a close code (4401 not authenticated, 4404 no such job), which a browser can read.
"""

from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.analysis_contract.models import JobState
from src.identity.domain.errors import IdentityError
from src.identity.domain.models import User
from src.ingestion.domain.job import to_progress

router = APIRouter(prefix="/ws", tags=["jobs"])

CLOSE_UNAUTHENTICATED = 4401
CLOSE_NOT_FOUND = 4404
_FINAL = (JobState.DONE, JobState.FAILED)


async def _authenticate(websocket: WebSocket, token: str | None) -> User | None:
    if token is None:
        return None
    container = websocket.app.state.container
    try:
        claims = await container.tokens.verify(token, "access")
    except IdentityError:
        return None
    user = await container.users.get_by_id(claims.user_id)
    return user if user is not None and user.is_active else None


@router.websocket("/jobs/{job_id}")
async def job_socket(websocket: WebSocket, job_id: UUID) -> None:
    offered = websocket.scope.get("subprotocols", [])
    token = offered[1] if len(offered) == 2 and offered[0] == "bearer" else None
    await websocket.accept(subprotocol="bearer" if token else None)

    user = await _authenticate(websocket, token)
    if user is None:
        await websocket.close(CLOSE_UNAUTHENTICATED)
        return
    ingestion = websocket.app.state.ingestion
    job = await ingestion.jobs.get(job_id)
    if job is None or job.owner_id != user.id:
        await websocket.close(CLOSE_NOT_FOUND)
        return

    try:
        # Subscribe before reading the snapshot, so no change can fall in between.
        async with ingestion.events.subscribe(job_id) as updates:
            snapshot = to_progress(await ingestion.jobs.get(job_id) or job)
            await websocket.send_text(snapshot.model_dump_json())
            if snapshot.state not in _FINAL:
                last = snapshot.progress
                async for progress in updates:
                    if progress.state not in _FINAL and progress.progress < last:
                        continue  # queued behind the snapshot: already superseded
                    last = progress.progress
                    await websocket.send_text(progress.model_dump_json())
        await websocket.close(1000)
    except WebSocketDisconnect:
        return
