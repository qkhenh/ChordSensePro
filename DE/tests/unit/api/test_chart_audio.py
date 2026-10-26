"""POST /charts/{id}/audio-ticket and GET|HEAD /charts/{id}/audio with HTTP Range (task 3.1)."""

from datetime import UTC, datetime
from uuid import UUID

import pytest

from src.analysis_contract.models import AnalysisResult, PlaybackSource, SourceKind
from src.ingestion.application.charts import build_chart
from src.ingestion.domain.job import new_job
from src.ingestion.infrastructure.mock_engine import FIXTURES_DIR
from tests.support.identity import PASSWORD

NOW = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)
SIZE = 10_000
DATA = bytes(i % 251 for i in range(SIZE))  # every byte position is recognisable


async def signup(api, email="ann@example.com") -> tuple[dict, UUID]:
    r = await api.client.post(
        "/api/v1/auth/register", json={"email": email, "password": PASSWORD, "name": "Ann"}
    )
    body = r.json()
    return {"Authorization": f"Bearer {body['access_token']}"}, UUID(body["user"]["id"])


async def add_chart(api, owner: UUID, *, playback=None, kind=SourceKind.FILE, with_file=True):
    result = AnalysisResult.model_validate_json(
        (FIXTURES_DIR / "ii_v_i_c.result.json").read_text(encoding="utf-8")
    )
    job = new_job(owner_id=owner, source_kind=kind, title="Song", now=NOW)
    chart = build_chart(job, result, now=NOW, playback=playback)
    await api.ingestion.charts.add(chart)
    if with_file and chart.source.playback.kind == "file":
        audio = api.data_dir / "audio"
        audio.mkdir(parents=True, exist_ok=True)
        (audio / f"{chart.source.content_hash}.wav").write_bytes(DATA)
    return chart


def audio_url(chart) -> str:
    return f"/api/v1/charts/{chart.id}/audio"


async def get_ticket_url(api, headers, chart) -> str:
    r = await api.client.post(f"{audio_url(chart)}-ticket", headers=headers)
    assert r.status_code == 200, r.text
    return r.json()["url"]


class TestFullAndRange:
    async def test_without_a_range_the_whole_file_comes_back(self, make_api):
        api = await make_api()
        headers, owner = await signup(api)
        chart = await add_chart(api, owner)
        r = await api.client.get(audio_url(chart), headers=headers)
        assert r.status_code == 200 and r.content == DATA
        assert r.headers["accept-ranges"] == "bytes"
        assert r.headers["content-type"] == "audio/wav"
        assert r.headers["content-length"] == str(SIZE)
        assert r.headers["cache-control"] == "private, max-age=3600"

    @pytest.mark.parametrize(
        ("header", "start", "end"),
        [
            ("bytes=0-99", 0, 99),
            ("bytes=9900-", 9900, 9999),
            ("bytes=-50", 9950, 9999),
            ("bytes=0-99999", 0, 9999),
            ("bytes=5-5", 5, 5),
            ("bytes=9999-9999", 9999, 9999),
            ("bytes=1234-5678", 1234, 5678),
            ("bytes=-99999", 0, 9999),
        ],
    )
    async def test_a_satisfiable_range_is_206_with_exactly_those_bytes(
        self, make_api, header, start, end
    ):
        api = await make_api()
        headers, owner = await signup(api)
        chart = await add_chart(api, owner)
        r = await api.client.get(audio_url(chart), headers={**headers, "Range": header})
        assert r.status_code == 206
        assert r.content == DATA[start : end + 1]
        assert r.headers["content-range"] == f"bytes {start}-{end}/{SIZE}"
        assert r.headers["content-length"] == str(end - start + 1)
        assert r.headers["accept-ranges"] == "bytes"

    @pytest.mark.parametrize("header", ["bytes=10000-", "bytes=20000-30000", "bytes=-0"])
    async def test_an_unsatisfiable_range_is_416_and_says_how_big_the_file_is(
        self, make_api, header
    ):
        api = await make_api()
        headers, owner = await signup(api)
        chart = await add_chart(api, owner)
        r = await api.client.get(audio_url(chart), headers={**headers, "Range": header})
        assert r.status_code == 416
        assert r.headers["content-range"] == f"bytes */{SIZE}"

    @pytest.mark.parametrize(
        "header",
        ["bytes=abc", "bytes=0-9,20-29", "bytes=9-5", "items=0-5", "bytes=-", "bytes=", "0-5"],
    )
    async def test_a_range_we_do_not_serve_is_ignored_and_the_whole_file_returned(
        self, make_api, header
    ):
        api = await make_api()
        headers, owner = await signup(api)
        chart = await add_chart(api, owner)
        r = await api.client.get(audio_url(chart), headers={**headers, "Range": header})
        assert r.status_code == 200 and r.content == DATA

    async def test_head_describes_the_body_without_sending_it(self, make_api):
        api = await make_api()
        headers, owner = await signup(api)
        chart = await add_chart(api, owner)
        plain = await api.client.head(audio_url(chart), headers=headers)
        assert plain.status_code == 200 and plain.content == b""
        assert plain.headers["content-length"] == str(SIZE)
        ranged = await api.client.head(audio_url(chart), headers={**headers, "Range": "bytes=0-99"})
        assert ranged.status_code == 206 and ranged.content == b""
        assert ranged.headers["content-range"] == f"bytes 0-99/{SIZE}"

    async def test_demo_charts_play_too(self, make_api):
        api = await make_api()
        headers, owner = await signup(api)
        chart = await add_chart(api, owner, kind=SourceKind.DEMO)
        assert (await api.client.get(audio_url(chart), headers=headers)).status_code == 200


class TestTickets:
    async def test_a_ticket_opens_the_audio_without_any_header(self, make_api):
        api = await make_api()
        headers, owner = await signup(api)
        chart = await add_chart(api, owner)
        url = await get_ticket_url(api, headers, chart)
        assert url.startswith(f"{audio_url(chart)}?ticket=")

        full = await api.client.get(url)  # exactly what an <audio> element does
        assert full.status_code == 200 and full.content == DATA
        part = await api.client.get(url, headers={"Range": "bytes=100-199"})
        assert part.status_code == 206 and part.content == DATA[100:200]

    async def test_the_ticket_response_says_when_it_stops_working(self, make_api):
        api = await make_api()
        headers, owner = await signup(api)
        chart = await add_chart(api, owner)
        body = (await api.client.post(f"{audio_url(chart)}-ticket", headers=headers)).json()
        assert set(body) == {"url", "expires_at"} and body["expires_at"].endswith("Z")

    async def test_an_expired_ticket_is_401_token_expired(self, make_api, clock):
        api = await make_api()
        headers, owner = await signup(api)
        chart = await add_chart(api, owner)
        url = await get_ticket_url(api, headers, chart)
        clock.advance(minutes=31)
        r = await api.client.get(url)
        assert (r.status_code, r.json()["code"]) == (401, "token_expired")

    async def test_a_ticket_for_one_chart_does_not_open_another(self, make_api):
        api = await make_api()
        headers, owner = await signup(api)
        first = await add_chart(api, owner)
        second = await add_chart(api, owner)
        ticket = (await get_ticket_url(api, headers, first)).split("ticket=")[1]
        r = await api.client.get(f"{audio_url(second)}?ticket={ticket}")
        assert (r.status_code, r.json()["code"]) == (401, "invalid_token")

    async def test_a_ticket_is_good_for_audio_and_nothing_else(self, make_api):
        api = await make_api()
        headers, owner = await signup(api)
        chart = await add_chart(api, owner)
        ticket = (await get_ticket_url(api, headers, chart)).split("ticket=")[1]
        for method, path in (
            ("GET", f"/api/v1/charts/{chart.id}?ticket={ticket}"),
            ("GET", f"/api/v1/charts?ticket={ticket}"),
            ("POST", f"{audio_url(chart)}-ticket?ticket={ticket}"),
            ("GET", f"/api/v1/users/me?ticket={ticket}"),
        ):
            r = await api.client.request(method, path)
            assert r.status_code == 401, path

    async def test_session_tokens_in_the_ticket_slot_are_refused(self, make_api):
        api = await make_api()
        r = await api.client.post(
            "/api/v1/auth/register",
            json={"email": "x@example.com", "password": PASSWORD, "name": "X"},
        )
        body = r.json()
        chart = await add_chart(api, UUID(body["user"]["id"]))
        for token in (body["access_token"], body["refresh_token"], "garbage"):
            r = await api.client.get(f"{audio_url(chart)}?ticket={token}")
            assert r.status_code == 401

    async def test_no_credentials_at_all_is_401(self, make_api):
        api = await make_api()
        _, owner = await signup(api)
        chart = await add_chart(api, owner)
        assert (await api.client.get(audio_url(chart))).status_code == 401
        assert (await api.client.post(f"{audio_url(chart)}-ticket")).status_code == 401


class TestWhoMayListen:
    async def test_someone_elses_chart_is_404_for_ticket_and_audio_alike(self, make_api):
        api = await make_api()
        _, ann = await signup(api, "ann@example.com")
        bob, _ = await signup(api, "bob@example.com")
        chart = await add_chart(api, ann)
        assert (await api.client.post(f"{audio_url(chart)}-ticket", headers=bob)).status_code == 404
        assert (await api.client.get(audio_url(chart), headers=bob)).status_code == 404

    async def test_unknown_charts_are_404_and_malformed_ids_422(self, make_api):
        api = await make_api()
        headers, _ = await signup(api)
        missing = "00000000-0000-4000-8000-000000000000"
        assert (
            await api.client.get(f"/api/v1/charts/{missing}/audio", headers=headers)
        ).status_code == 404
        assert (
            await api.client.get("/api/v1/charts/not-a-uuid/audio", headers=headers)
        ).status_code == 422

    @pytest.mark.parametrize(
        "playback",
        [
            PlaybackSource(kind="youtube", embed_id="dQw4w9WgXcQ"),
            PlaybackSource(kind="soundcloud", embed_id="https://soundcloud.com/a/b"),
            PlaybackSource(kind="direct", audio_url="https://cdn.example.com/a.mp3"),
        ],
    )
    async def test_charts_without_stored_audio_have_none_to_serve(self, make_api, playback):
        api = await make_api()
        headers, owner = await signup(api)
        chart = await add_chart(api, owner, playback=playback, kind=SourceKind.URL)
        assert (
            await api.client.post(f"{audio_url(chart)}-ticket", headers=headers)
        ).status_code == 404
        assert (await api.client.get(audio_url(chart), headers=headers)).status_code == 404

    async def test_a_chart_whose_file_has_gone_is_404_not_a_crash(self, make_api):
        api = await make_api()
        headers, owner = await signup(api)
        chart = await add_chart(api, owner, with_file=False)
        assert (
            await api.client.post(f"{audio_url(chart)}-ticket", headers=headers)
        ).status_code == 404
        assert (await api.client.get(audio_url(chart), headers=headers)).status_code == 404

    async def test_a_file_that_vanishes_after_the_ticket_was_issued_is_404(self, make_api):
        api = await make_api()
        headers, owner = await signup(api)
        chart = await add_chart(api, owner)
        url = await get_ticket_url(api, headers, chart)
        for kept in (api.data_dir / "audio").glob("*.wav"):
            kept.unlink()
        assert (await api.client.get(url)).status_code == 404
