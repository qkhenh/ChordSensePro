"""WS /api/v1/ws/jobs/{id}: snapshot first, then live changes, authenticated by subprotocol."""

import time
from uuid import UUID

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from src.analysis_contract.models import JobStage, JobState, SourceKind
from src.api.app import create_app
from src.api.settings import ApiSettings
from src.ingestion.domain.job import advance, new_job, to_progress
from src.ingestion.infrastructure.ffmpeg import FfmpegNormalizer
from tests.support.identity import PASSWORD, FakeClock, make_container
from tests.support.ingestion import build_ingestion


class Env:
    def __init__(self, tmp_path):
        self.clock = FakeClock()
        self.container = make_container(self.clock)
        self.ingestion = build_ingestion(
            tmp_path / "data",
            quota=self.container.quota,
            clock=self.clock,
            normalizer=FfmpegNormalizer(),
        )
        self.app = create_app(self.container, ApiSettings(), ingestion=self.ingestion)

    def signup(self, client, email="ann@example.com") -> tuple[str, UUID]:
        r = client.post(
            "/api/v1/auth/register", json={"email": email, "password": PASSWORD, "name": "Ann"}
        )
        body = r.json()
        return body["access_token"], UUID(body["user"]["id"])

    def new_job(self, client, owner: UUID):
        job = new_job(owner_id=owner, source_kind=SourceKind.FILE, title="Song", now=self.clock())
        return client.portal.call(self.ingestion.jobs.add, job)

    def move(self, client, job, **changes):
        moved = advance(job, now=self.clock(), **changes)

        async def apply():
            await self.ingestion.jobs.save(moved)
            await self.ingestion.events.publish(to_progress(moved))

        client.portal.call(apply)
        return moved

    def subscribers(self, client, job_id) -> int:
        return self.ingestion.events.subscriber_count(job_id)


@pytest.fixture
def env(tmp_path):
    return Env(tmp_path)


def url(job_id) -> str:
    return f"/api/v1/ws/jobs/{job_id}"


def expect_close(client, job_id, subprotocols, code):
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(url(job_id), subprotocols=subprotocols) as ws:
            ws.receive_json()
    assert exc.value.code == code


class TestStreaming:
    def test_sends_the_current_state_then_every_change_then_closes(self, env):
        with TestClient(env.app) as client:
            token, owner = env.signup(client)
            job = env.new_job(client, owner)
            with client.websocket_connect(url(job.id), subprotocols=["bearer", token]) as ws:
                assert ws.accepted_subprotocol == "bearer"
                assert ws.receive_json()["state"] == "queued"

                job = env.move(
                    client,
                    job,
                    state=JobState.PROCESSING,
                    stage=JobStage.NORMALIZING,
                    progress=0.15,
                )
                assert ws.receive_json()["stage"] == "normalizing"
                job = env.move(client, job, state=JobState.DONE, chart_id=UUID(int=7))
                last = ws.receive_json()
                assert (last["state"], last["progress"]) == ("done", 1.0)
                with pytest.raises(WebSocketDisconnect) as exc:
                    ws.receive_json()
                assert exc.value.code == 1000

    def test_a_job_that_already_finished_sends_one_snapshot_and_closes(self, env):
        with TestClient(env.app) as client:
            token, owner = env.signup(client)
            job = env.new_job(client, owner)
            env.move(client, job, state=JobState.DONE, chart_id=UUID(int=7))
            with client.websocket_connect(url(job.id), subprotocols=["bearer", token]) as ws:
                assert ws.receive_json()["state"] == "done"
                with pytest.raises(WebSocketDisconnect) as exc:
                    ws.receive_json()
                assert exc.value.code == 1000

    def test_the_wire_format_equals_the_polling_endpoint(self, env):
        with TestClient(env.app) as client:
            token, owner = env.signup(client)
            job = env.new_job(client, owner)
            polled = client.get(
                f"/api/v1/analyze/status/{job.id}", headers={"Authorization": f"Bearer {token}"}
            ).json()
            with client.websocket_connect(url(job.id), subprotocols=["bearer", token]) as ws:
                assert ws.receive_json() == polled

    def test_a_stale_event_after_the_snapshot_is_not_replayed(self, env):
        with TestClient(env.app) as client:
            token, owner = env.signup(client)
            job = env.new_job(client, owner)
            job = env.move(
                client, job, state=JobState.PROCESSING, stage=JobStage.SEPARATING, progress=0.45
            )
            with client.websocket_connect(url(job.id), subprotocols=["bearer", token]) as ws:
                assert ws.receive_json()["progress"] == 0.45
                stale = to_progress(job).model_copy(update={"progress": 0.15})
                client.portal.call(env.ingestion.events.publish, stale)
                env.move(
                    client,
                    job,
                    state=JobState.PROCESSING,
                    stage=JobStage.BEAT_TRACKING,
                    progress=0.7,
                )
                assert ws.receive_json()["progress"] == 0.7

    def test_a_change_reaches_the_client_within_a_second(self, env):
        with TestClient(env.app) as client:
            token, owner = env.signup(client)
            job = env.new_job(client, owner)
            with client.websocket_connect(url(job.id), subprotocols=["bearer", token]) as ws:
                ws.receive_json()
                started = time.perf_counter()
                env.move(
                    client,
                    job,
                    state=JobState.PROCESSING,
                    stage=JobStage.NORMALIZING,
                    progress=0.15,
                )
                ws.receive_json()
                assert time.perf_counter() - started < 1.0

    def test_a_dropped_client_is_unsubscribed(self, env):
        with TestClient(env.app) as client:
            token, owner = env.signup(client)
            job = env.new_job(client, owner)
            with client.websocket_connect(url(job.id), subprotocols=["bearer", token]) as ws:
                ws.receive_json()
                assert env.subscribers(client, job.id) == 1
            deadline = time.time() + 2
            while env.subscribers(client, job.id) and time.time() < deadline:
                time.sleep(0.02)
            assert env.subscribers(client, job.id) == 0


class TestRefusals:
    def test_no_credentials_closes_4401(self, env):
        with TestClient(env.app) as client:
            _, owner = env.signup(client)
            job = env.new_job(client, owner)
            expect_close(client, job.id, None, 4401)

    def test_a_forged_token_closes_4401(self, env):
        with TestClient(env.app) as client:
            _, owner = env.signup(client)
            job = env.new_job(client, owner)
            expect_close(client, job.id, ["bearer", "not.a.jwt"], 4401)

    def test_an_expired_token_closes_4401(self, env):
        with TestClient(env.app) as client:
            token, owner = env.signup(client)
            job = env.new_job(client, owner)
            env.clock.advance(minutes=16)  # access tokens live 15 minutes
            expect_close(client, job.id, ["bearer", token], 4401)

    def test_a_refresh_token_is_not_accepted_as_an_access_token(self, env):
        with TestClient(env.app) as client:
            r = client.post(
                "/api/v1/auth/register",
                json={"email": "x@example.com", "password": PASSWORD, "name": "X"},
            ).json()
            job = env.new_job(client, UUID(r["user"]["id"]))
            expect_close(client, job.id, ["bearer", r["refresh_token"]], 4401)

    def test_someone_elses_job_and_unknown_jobs_close_4404(self, env):
        with TestClient(env.app) as client:
            _, ann = env.signup(client, "ann@example.com")
            bob_token, _ = env.signup(client, "bob@example.com")
            job = env.new_job(client, ann)
            expect_close(client, job.id, ["bearer", bob_token], 4404)
            expect_close(client, UUID(int=99), ["bearer", bob_token], 4404)
