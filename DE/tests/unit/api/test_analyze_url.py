"""POST /api/v1/analyze/url (task 2.2): the SSRF guard sits in front of quota and queue."""

import pytest

from tests.support.identity import PASSWORD

URL = "/api/v1/analyze/url"
QUOTA = "/api/v1/users/me/quota"
PUBLIC_URL = "http://93.184.216.34/music/Take%20Five.mp3"


async def signup(api, email="ann@example.com") -> dict:
    r = await api.client.post(
        "/api/v1/auth/register", json={"email": email, "password": PASSWORD, "name": "Ann"}
    )
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


async def submit(api, headers, url):
    return await api.client.post(URL, headers=headers, json={"url": url})


async def used(api, headers) -> int:
    return (await api.client.get(QUOTA, headers=headers)).json()["used"]


async def finished(api, headers, job_id) -> dict:
    await api.ingestion.runner.drain()
    return (await api.client.get(f"/api/v1/analyze/status/{job_id}", headers=headers)).json()


class TestBlocked:
    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1/x.mp3",
            "http://169.254.169.254/latest/meta-data/",
            "http://localhost/x.mp3",
            "http://[::1]/x.mp3",
            "http://2130706433/x.mp3",
            "file:///etc/passwd",
            "ftp://example.com/x.mp3",
            "http://user:pass@example.com/x.mp3",
            "http://example.com:22/x.mp3",
        ],
    )
    async def test_unsafe_links_are_422_ssrf_blocked_and_cost_nothing(self, make_api, url):
        api = await make_api()
        headers = await signup(api)
        r = await submit(api, headers, url)
        assert r.status_code == 422
        body = r.json()
        assert body["code"] == "ssrf_blocked" and body["message"]
        assert await used(api, headers) == 0
        assert await api.ingestion.jobs.list_unfinished() == []

    async def test_a_hostname_that_resolves_to_a_private_address_is_blocked(self, make_api):
        api = await make_api()
        headers = await signup(api)

        async def to_private(host, port):
            return ["10.0.0.5"]

        api.ingestion.submission._resolve = to_private
        r = await submit(api, headers, "https://rebind.example/x.mp3")
        assert (r.status_code, r.json()["code"]) == (422, "ssrf_blocked")


class TestAccepted:
    async def test_a_public_link_runs_through_fetching_to_a_chart(self, make_api):
        api = await make_api()
        headers = await signup(api)
        r = await submit(api, headers, PUBLIC_URL)
        assert r.status_code == 202

        status = await finished(api, headers, r.json()["job_id"])
        assert status["state"] == "done"
        chart = (
            await api.client.get(f"/api/v1/charts/{status['result_chart_id']}", headers=headers)
        ).json()
        assert chart["title"] == "Take Five"
        assert chart["source"]["kind"] == "url" and chart["source"]["url"] == PUBLIC_URL
        assert chart["source"]["playback"] == {
            "kind": "direct",
            "audio_url": PUBLIC_URL,
            "embed_id": None,
        }
        assert await used(api, headers) == 1

    async def test_a_youtube_link_is_played_by_the_embedded_player(self, make_api):
        api = await make_api()
        headers = await signup(api)
        r = await submit(api, headers, "https://youtu.be/dQw4w9WgXcQ")
        status = await finished(api, headers, r.json()["job_id"])
        chart = (
            await api.client.get(f"/api/v1/charts/{status['result_chart_id']}", headers=headers)
        ).json()
        assert chart["source"]["playback"]["kind"] == "youtube"
        assert chart["source"]["playback"]["embed_id"] == "dQw4w9WgXcQ"

    async def test_an_unresolvable_link_fails_the_job_and_refunds(self, make_api):
        api = await make_api()
        headers = await signup(api)
        r = await submit(api, headers, "https://example.com/fail-unresolvable.mp3")
        assert r.status_code == 202
        status = await finished(api, headers, r.json()["job_id"])
        assert status["state"] == "failed"
        assert status["error"]["code"] == "url_unresolvable"
        assert await used(api, headers) == 0

    async def test_the_sixth_link_of_the_day_is_429(self, make_api):
        api = await make_api()
        headers = await signup(api)
        for i in range(5):
            assert (await submit(api, headers, f"{PUBLIC_URL}?n={i}")).status_code == 202
        r = await submit(api, headers, PUBLIC_URL)
        assert (r.status_code, r.json()["code"]) == (429, "quota_exceeded")


class TestBody:
    async def test_the_url_field_is_required_and_nothing_else_is_accepted(self, make_api):
        api = await make_api()
        headers = await signup(api)
        for body in ({}, {"url": ""}, {"url": PUBLIC_URL, "extra": 1}, {"url": "x" * 3000}):
            r = await api.client.post(URL, headers=headers, json=body)
            assert (r.status_code, r.json()["code"]) == (422, "validation_error"), body

    async def test_requires_a_signed_in_user(self, make_api):
        api = await make_api()
        assert (await api.client.post(URL, json={"url": PUBLIC_URL})).status_code == 401
