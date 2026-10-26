"""POST /api/v1/analyze/file, GET /analyze/status/{id}, GET /charts/{id} (tasks 2.1, 2.3)."""

import time
import uuid

import pytest

from tests.support.audio import exact_size_wav, needs_ffmpeg, write_wav
from tests.support.identity import PASSWORD

pytestmark = needs_ffmpeg

FILE = "/api/v1/analyze/file"
QUOTA = "/api/v1/users/me/quota"


async def signup(api, email="ann@example.com") -> dict:
    r = await api.client.post(
        "/api/v1/auth/register", json={"email": email, "password": PASSWORD, "name": "Ann"}
    )
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


async def upload(api, headers, name="My Song.wav", data: bytes | None = None, field="file"):
    if data is None:
        data = exact_size_wav(8044)
    return await api.client.post(FILE, headers=headers, files={field: (name, data, "audio/wav")})


async def used(api, headers) -> int:
    return (await api.client.get(QUOTA, headers=headers)).json()["used"]


def leftovers(api) -> list:
    tmp = api.data_dir / "tmp"
    return list(tmp.iterdir()) if tmp.exists() else []


class TestAccepting:
    async def test_a_valid_wav_returns_202_and_runs_to_a_chart(self, make_api, tmp_path):
        api = await make_api()
        headers = await signup(api)
        wav = write_wav(tmp_path / "in.wav", seconds=1.0).read_bytes()

        r = await upload(api, headers, "My Song.wav", wav)
        assert r.status_code == 202
        job_id = r.json()["job_id"]
        uuid.UUID(job_id)

        await api.ingestion.runner.drain()
        status = (await api.client.get(f"/api/v1/analyze/status/{job_id}", headers=headers)).json()
        assert (status["state"], status["progress"]) == ("done", 1.0)

        chart = await api.client.get(f"/api/v1/charts/{status['result_chart_id']}", headers=headers)
        assert chart.status_code == 200
        body = chart.json()
        assert body["title"] == "My Song"
        assert body["ai_events"] and body["overrides"] == []
        assert body["source"]["kind"] == "file"
        assert body["source"]["playback"]["kind"] == "file"
        assert await used(api, headers) == 1

    async def test_requires_a_signed_in_user(self, make_api):
        api = await make_api()
        r = await api.client.post(
            FILE, files={"file": ("a.wav", exact_size_wav(8044), "audio/wav")}
        )
        assert r.status_code == 401

    async def test_a_missing_file_field_is_a_validation_error(self, make_api):
        api = await make_api()
        headers = await signup(api)
        r = await upload(api, headers, field="not_file")
        assert r.status_code == 422
        assert r.json()["code"] == "validation_error"


class TestRejecting:
    @pytest.mark.parametrize(
        ("name", "data"),
        [
            ("fake.mp3", b"this is plain text, not audio"),
            ("empty.wav", b""),
            ("picture.wav", b"\x89PNG\r\n\x1a\n" + b"\x00" * 64),
            ("noaudio.mp3", b"ID3" + b"\x00" * 100),
        ],
    )
    async def test_non_audio_is_415_and_costs_nothing(self, make_api, name, data):
        api = await make_api()
        headers = await signup(api)
        r = await upload(api, headers, name, data)
        assert r.status_code == 415
        assert r.json()["code"] == "unsupported_format"
        assert await used(api, headers) == 0
        assert leftovers(api) == []
        assert await api.ingestion.jobs.list_unfinished() == []

    async def test_a_file_of_exactly_the_limit_is_accepted_and_one_byte_more_is_413(self, make_api):
        api = await make_api(max_upload_bytes=4096)
        headers = await signup(api)
        assert (await upload(api, headers, data=exact_size_wav(4096))).status_code == 202

        over = await upload(api, headers, data=exact_size_wav(4096) + b"\x00")
        assert over.status_code == 413
        assert over.json()["code"] == "file_too_large"
        assert await used(api, headers) == 1  # only the accepted upload was charged
        await api.ingestion.runner.drain()
        assert leftovers(api) == []

    async def test_duration_boundary_is_ten_minutes_exactly(self, make_api):
        api = await make_api()
        headers = await signup(api)
        rate = 8000
        ten_minutes = rate * 2 * 600  # bytes of mono 16-bit payload
        ok = await upload(api, headers, data=exact_size_wav(44 + ten_minutes, rate))
        assert ok.status_code == 202

        over = await upload(api, headers, data=exact_size_wav(44 + ten_minutes + 16, rate))
        assert over.status_code == 413  # 600.001 s
        assert over.json()["code"] == "duration_too_long"
        assert await used(api, headers) == 1

    async def test_an_oversized_stream_is_cut_off_before_it_is_fully_read(self, make_api):
        api = await make_api(max_upload_bytes=4096)
        headers = await signup(api)
        big = exact_size_wav(4096 + 200_000)
        r = await upload(api, headers, data=big)
        assert r.status_code == 413
        assert r.json()["code"] == "file_too_large"
        assert leftovers(api) == []

    async def test_hostile_file_names_never_leave_the_work_directory(self, make_api):
        api = await make_api()
        headers = await signup(api)
        for name in ("../../evil.wav", r"..\..\evil.wav", "x" * 1000 + ".wav"):
            r = await upload(api, headers, name)
            assert r.status_code == 202, name
        await api.ingestion.runner.drain()
        assert not (api.data_dir.parent / "evil.wav").exists()
        titles = {job.title for job in api.ingestion.jobs._jobs.values()}
        assert titles == {"evil", "x" * 300}


class TestQuota:
    async def test_five_uploads_pass_and_the_sixth_is_429_with_the_reset_time(self, make_api):
        api = await make_api()
        headers = await signup(api)
        for i in range(5):  # different audio each time: identical audio is a cache hit
            r = await upload(api, headers, f"s{i}.wav", exact_size_wav(8044, fill=i + 1))
            assert r.status_code == 202
        r = await upload(api, headers, "sixth.wav", exact_size_wav(8044, fill=9))
        assert r.status_code == 429
        body = r.json()
        assert body["code"] == "quota_exceeded" and body["limit"] == 5
        assert body["resets_at"].endswith("Z")
        assert int(r.headers["retry-after"]) > 0
        assert await used(api, headers) == 5

    async def test_a_failed_start_gives_the_quota_back(self, make_api):
        api = await make_api()
        headers = await signup(api)

        class Broken:
            pending = 0

            async def enqueue(self, job_id):
                raise RuntimeError("queue is down")

        api.ingestion.submission._runner = Broken()
        r = await upload(api, headers)
        assert r.status_code == 502
        assert r.json()["code"] == "ai_failure"
        assert await used(api, headers) == 0
        assert leftovers(api) == []
        (job,) = api.ingestion.jobs._jobs.values()
        assert job.state.value == "failed" and job.error.retryable is True


class TestOwnership:
    async def test_other_users_cannot_see_a_job_or_a_chart(self, make_api):
        api = await make_api()
        ann = await signup(api, "ann@example.com")
        bob = await signup(api, "bob@example.com")
        job_id = (await upload(api, ann)).json()["job_id"]
        await api.ingestion.runner.drain()
        chart_id = (await api.client.get(f"/api/v1/analyze/status/{job_id}", headers=ann)).json()[
            "result_chart_id"
        ]

        for url in (f"/api/v1/analyze/status/{job_id}", f"/api/v1/charts/{chart_id}"):
            assert (await api.client.get(url, headers=bob)).status_code == 404
            assert (await api.client.get(url, headers=ann)).status_code == 200

    async def test_unknown_ids_are_404_and_malformed_ids_422(self, make_api):
        api = await make_api()
        headers = await signup(api)
        for path in ("analyze/status", "charts"):
            missing = await api.client.get(f"/api/v1/{path}/{uuid.uuid4()}", headers=headers)
            assert missing.status_code == 404 and missing.json()["code"] == "not_found"
            bad = await api.client.get(f"/api/v1/{path}/not-a-uuid", headers=headers)
            assert bad.status_code == 422

    async def test_status_and_charts_need_authentication(self, make_api):
        api = await make_api()
        for path in ("analyze/status", "charts"):
            r = await api.client.get(f"/api/v1/{path}/{uuid.uuid4()}")
            assert r.status_code == 401


class TestLatency:
    async def test_accepting_a_job_takes_at_most_300_ms_at_p95(self, make_api):
        api = await make_api(free_quota=1000)
        headers = await signup(api)
        timings = []
        for i in range(50):
            started = time.perf_counter()
            r = await upload(api, headers, f"s{i}.wav", exact_size_wav(100_044, fill=i % 200))
            timings.append(time.perf_counter() - started)
            assert r.status_code == 202
        timings.sort()
        assert timings[int(len(timings) * 0.95) - 1] <= 0.3, timings[-5:]
