"""Quota and cache behaviour end to end over HTTP (task 2.4)."""

import pytest

from tests.support.audio import encode, exact_size_wav, needs_ffmpeg, write_wav
from tests.support.identity import PASSWORD

pytestmark = needs_ffmpeg

FILE = "/api/v1/analyze/file"
QUOTA = "/api/v1/users/me/quota"


async def signup(api, email="ann@example.com") -> dict:
    r = await api.client.post(
        "/api/v1/auth/register", json={"email": email, "password": PASSWORD, "name": "Ann"}
    )
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


async def upload(api, headers, name, data):
    return await api.client.post(FILE, headers=headers, files={"file": (name, data, "audio/wav")})


async def used(api, headers) -> int:
    return (await api.client.get(QUOTA, headers=headers)).json()["used"]


async def status(api, headers, response) -> dict:
    job_id = response.json()["job_id"]
    return (await api.client.get(f"/api/v1/analyze/status/{job_id}", headers=headers)).json()


def song(n: int) -> bytes:
    return exact_size_wav(8044, fill=n)


class TestCacheByUploadedBytes:
    async def test_the_same_file_again_is_served_from_the_cache_without_quota(self, make_api):
        api = await make_api()
        headers = await signup(api)
        first = await upload(api, headers, "Song One.wav", song(1))
        await api.ingestion.runner.drain()
        for n in range(2, 6):
            assert (await upload(api, headers, f"s{n}.wav", song(n))).status_code == 202
        await api.ingestion.runner.drain()
        assert await used(api, headers) == 5  # the daily allowance is spent

        again = await upload(api, headers, "Song One again.wav", song(1))
        assert again.status_code == 202  # not 429: nothing new is analysed
        done = await status(api, headers, again)
        assert (done["state"], done["progress"]) == ("done", 1.0)
        assert done["result_chart_id"] != (await status(api, headers, first))["result_chart_id"]
        assert await used(api, headers) == 5

        assert (await upload(api, headers, "new.wav", song(9))).status_code == 429

    async def test_another_user_gets_their_own_chart_without_spending_quota(self, make_api):
        api = await make_api()
        ann = await signup(api, "ann@example.com")
        bob = await signup(api, "bob@example.com")
        first = await upload(api, ann, "Shared.wav", song(1))
        await api.ingestion.runner.drain()
        ann_chart = (await status(api, ann, first))["result_chart_id"]

        second = await upload(api, bob, "Shared.wav", song(1))
        bob_chart = (await status(api, bob, second))["result_chart_id"]
        assert bob_chart != ann_chart
        assert (await api.client.get(f"/api/v1/charts/{bob_chart}", headers=bob)).status_code == 200
        assert (await api.client.get(f"/api/v1/charts/{bob_chart}", headers=ann)).status_code == 404
        assert await used(api, bob) == 0

    async def test_the_cached_chart_equals_the_analysed_one(self, make_api):
        api = await make_api()
        headers = await signup(api)
        first = await upload(api, headers, "Song.wav", song(1))
        await api.ingestion.runner.drain()
        again = await upload(api, headers, "Song.wav", song(1))
        charts = []
        for response in (first, again):
            chart_id = (await status(api, headers, response))["result_chart_id"]
            charts.append(
                (await api.client.get(f"/api/v1/charts/{chart_id}", headers=headers)).json()
            )
        assert charts[0]["ai_events"] == charts[1]["ai_events"]
        assert charts[0]["source"]["content_hash"] == charts[1]["source"]["content_hash"]

    async def test_without_the_kept_audio_the_cache_cannot_short_circuit(self, make_api):
        api = await make_api()
        headers = await signup(api)
        await upload(api, headers, "Song.wav", song(1))
        await api.ingestion.runner.drain()
        for kept in (api.data_dir / "audio").glob("*.wav"):
            kept.unlink()  # a chart without playable audio would be useless

        again = await upload(api, headers, "Song.wav", song(1))
        assert (await status(api, headers, again))["state"] != "done"  # goes through the queue
        assert await used(api, headers) == 2
        await api.ingestion.runner.drain()
        assert (await status(api, headers, again))["state"] == "done"
        assert await used(api, headers) == 1  # same content: the second charge came back
        assert list((api.data_dir / "audio").glob("*.wav"))  # and the audio is kept again

    async def test_no_temporary_files_are_left_by_a_cache_hit(self, make_api):
        api = await make_api()
        headers = await signup(api)
        await upload(api, headers, "Song.wav", song(1))
        await api.ingestion.runner.drain()
        await upload(api, headers, "Song.wav", song(1))
        await api.ingestion.runner.drain()
        tmp = api.data_dir / "tmp"
        assert not tmp.exists() or list(tmp.iterdir()) == []


class TestCacheByWaveform:
    async def test_the_same_music_in_another_encoding_is_charged_then_refunded(
        self, make_api, tmp_path
    ):
        api = await make_api()
        headers = await signup(api)
        wav = write_wav(tmp_path / "s.wav", seconds=1.0)
        flac = encode(wav, tmp_path / "s.flac")

        await upload(api, headers, "Song.wav", wav.read_bytes())
        await api.ingestion.runner.drain()
        assert await used(api, headers) == 1

        second = await upload(api, headers, "Song.flac", flac.read_bytes())
        assert second.status_code == 202
        assert await used(api, headers) == 2  # different bytes: only the waveform tells
        await api.ingestion.runner.drain()
        assert (await status(api, headers, second))["state"] == "done"
        assert await used(api, headers) == 1


class TestFailureRefund:
    async def test_a_failed_analysis_gives_the_quota_back(self, make_api):
        api = await make_api()
        headers = await signup(api)
        response = await upload(api, headers, "fail-ai.wav", song(3))
        assert await used(api, headers) == 1
        await api.ingestion.runner.drain()
        failed = await status(api, headers, response)
        assert failed["state"] == "failed" and failed["error"]["code"] == "ai_failure"
        assert await used(api, headers) == 0

    async def test_a_timeout_gives_the_quota_back(self, make_api):
        api = await make_api()
        api.ingestion.executor._settings = api.ingestion.settings.model_copy(
            update={"job_timeout_s": 0.2}
        )
        headers = await signup(api)
        response = await upload(api, headers, "fail-timeout.wav", song(4))
        await api.ingestion.runner.drain()
        failed = await status(api, headers, response)
        assert failed["error"]["code"] == "analysis_timeout"
        assert await used(api, headers) == 0

    @pytest.mark.parametrize("role", ["paid"])
    async def test_paid_users_are_never_charged_or_refunded(self, make_api, role):
        api = await make_api()
        headers = await signup(api)
        user = await api.container.users.get_by_email("ann@example.com")
        await api.container.users.set_role(user.id, role)
        for n in range(1, 8):  # past the free allowance
            assert (await upload(api, headers, f"s{n}.wav", song(n))).status_code == 202
        await api.ingestion.runner.drain()
        assert await used(api, headers) == 0
