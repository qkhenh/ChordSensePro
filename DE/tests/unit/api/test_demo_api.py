"""GET /demo-tracks and POST /analyze/demo/{id} (task 2.7): free, and never charged."""

import json

import pytest

from src.ingestion.application.demo_audio import generate_demo_audio
from src.ingestion.infrastructure.mock_engine import FIXTURES_DIR
from tests.support.audio import exact_size_wav, needs_ffmpeg
from tests.support.identity import PASSWORD

pytestmark = needs_ffmpeg

TRACKS = "/api/v1/demo-tracks"
QUOTA = "/api/v1/users/me/quota"


async def signup(api, email="ann@example.com") -> dict:
    r = await api.client.post(
        "/api/v1/auth/register", json={"email": email, "password": PASSWORD, "name": "Ann"}
    )
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


async def used(api, headers) -> int:
    return (await api.client.get(QUOTA, headers=headers)).json()["used"]


class TestCatalogEndpoint:
    async def test_lists_the_seven_demo_tracks(self, make_api):
        api = await make_api()
        headers = await signup(api)
        r = await api.client.get(TRACKS, headers=headers)
        assert r.status_code == 200
        tracks = r.json()
        assert len(tracks) == 7
        first = tracks[0]
        assert set(first) == {"id", "title", "artist", "duration_ms", "tempo_bpm", "key"}
        assert first["id"] == "ii_v_i_c" and first["key"] == "C major"

    async def test_requires_a_signed_in_user(self, make_api):
        api = await make_api()
        assert (await api.client.get(TRACKS)).status_code == 401
        assert (await api.client.post("/api/v1/analyze/demo/ii_v_i_c")).status_code == 401


class TestAnalyzeDemo:
    async def test_runs_the_chosen_song_and_returns_its_golden_chart(self, make_api):
        api = await make_api()
        generate_demo_audio(api.ingestion.demo, ["ii_v_i_c"])
        headers = await signup(api)

        r = await api.client.post("/api/v1/analyze/demo/ii_v_i_c", headers=headers)
        assert r.status_code == 202
        await api.ingestion.runner.drain()
        status = (
            await api.client.get(f"/api/v1/analyze/status/{r.json()['job_id']}", headers=headers)
        ).json()
        assert status["state"] == "done"

        chart = (
            await api.client.get(f"/api/v1/charts/{status['result_chart_id']}", headers=headers)
        ).json()
        golden = json.loads((FIXTURES_DIR / "ii_v_i_c.result.json").read_text("utf-8"))
        assert chart["source"]["kind"] == "demo"
        assert chart["source"]["playback"]["kind"] == "file"
        assert [e["symbol"] for e in chart["ai_events"]] == [
            e["symbol"] for e in golden["chord_events"]
        ]
        assert chart["title"] == next(
            t["title"]
            for t in (await api.client.get(TRACKS, headers=headers)).json()
            if t["id"] == "ii_v_i_c"
        )

    async def test_costs_no_quota_even_when_the_allowance_is_spent(self, make_api):
        api = await make_api()
        generate_demo_audio(api.ingestion.demo, ["ii_v_i_c"])
        headers = await signup(api)
        for n in range(1, 6):
            files = {"file": (f"s{n}.wav", exact_size_wav(8044, fill=n), "audio/wav")}
            await api.client.post("/api/v1/analyze/file", headers=headers, files=files)
        await api.ingestion.runner.drain()
        assert await used(api, headers) == 5

        r = await api.client.post("/api/v1/analyze/demo/ii_v_i_c", headers=headers)
        assert r.status_code == 202
        await api.ingestion.runner.drain()
        assert await used(api, headers) == 5

    async def test_a_second_run_of_the_same_demo_also_works(self, make_api):
        api = await make_api()
        generate_demo_audio(api.ingestion.demo, ["ii_v_i_c"])
        headers = await signup(api)
        states = []
        for _ in range(2):
            r = await api.client.post("/api/v1/analyze/demo/ii_v_i_c", headers=headers)
            await api.ingestion.runner.drain()
            job = r.json()["job_id"]
            states.append(
                (await api.client.get(f"/api/v1/analyze/status/{job}", headers=headers)).json()[
                    "state"
                ]
            )
        assert states == ["done", "done"]

    @pytest.mark.parametrize("bad", ["nope", "..%2Fii_v_i_c", "ii_v_i_c.wav", "II_V_I_C"])
    async def test_unknown_ids_are_404(self, make_api, bad):
        api = await make_api()
        headers = await signup(api)
        r = await api.client.post(f"/api/v1/analyze/demo/{bad}", headers=headers)
        assert (r.status_code, r.json()["code"]) == (404, "not_found")

    async def test_missing_audio_is_a_clear_503_not_a_crash(self, make_api):
        api = await make_api()  # the generator script has not been run
        headers = await signup(api)
        r = await api.client.post("/api/v1/analyze/demo/ii_v_i_c", headers=headers)
        assert (r.status_code, r.json()["code"]) == (503, "demo_unavailable")
        assert await api.ingestion.jobs.list_unfinished() == []
