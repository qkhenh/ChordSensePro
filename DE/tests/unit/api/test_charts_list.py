"""GET /api/v1/charts (keyset pagination) and ETag on GET /charts/{id} (task 3.1)."""

from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from src.analysis_contract.models import AnalysisResult, Chart, SourceKind
from src.ingestion.application.charts import build_chart
from src.ingestion.domain.job import new_job
from src.ingestion.infrastructure.mock_engine import FIXTURES_DIR
from tests.support.identity import PASSWORD

LIST = "/api/v1/charts"
T0 = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)


async def signup(api, email="ann@example.com") -> tuple[dict, UUID]:
    r = await api.client.post(
        "/api/v1/auth/register", json={"email": email, "password": PASSWORD, "name": "Ann"}
    )
    body = r.json()
    return {"Authorization": f"Bearer {body['access_token']}"}, UUID(body["user"]["id"])


def result(name: str = "ii_v_i_c") -> AnalysisResult:
    return AnalysisResult.model_validate_json(
        (FIXTURES_DIR / f"{name}.result.json").read_text(encoding="utf-8")
    )


async def add_chart(
    api, owner: UUID, *, title="Song", age_s: int = 0, version: int = 1, name="ii_v_i_c"
) -> Chart:
    job = new_job(owner_id=owner, source_kind=SourceKind.FILE, title=title, now=T0)
    chart = build_chart(job, result(name), now=T0).model_copy(
        update={"updated_at": T0 - timedelta(seconds=age_s), "version": version}
    )
    await api.ingestion.charts.add(chart)
    return chart


class TestList:
    async def test_newest_first(self, make_api):
        api = await make_api()
        headers, owner = await signup(api)
        old = await add_chart(api, owner, title="Old", age_s=300)
        new = await add_chart(api, owner, title="New", age_s=10)
        mid = await add_chart(api, owner, title="Mid", age_s=100)

        page = (await api.client.get(LIST, headers=headers)).json()
        assert [i["id"] for i in page["items"]] == [str(new.id), str(mid.id), str(old.id)]
        assert page["next_cursor"] is None

    async def test_each_item_is_a_light_summary_with_the_numbers_a_library_needs(self, make_api):
        api = await make_api()
        headers, owner = await signup(api)
        chart = await add_chart(api, owner, title="Take Five", name="blues_f")

        (item,) = (await api.client.get(LIST, headers=headers)).json()["items"]
        assert item == {
            "id": str(chart.id),
            "title": "Take Five",
            "artist": None,
            "source_kind": "file",
            "playback_kind": "file",
            "duration_ms": 30150,
            "tempo_bpm": 100.0,
            "key": "F major",
            "is_customized": False,
            "version": 1,
            "updated_at": item["updated_at"],
        }
        assert "ai_events" not in item  # the heavy part stays out of the list

    async def test_pages_cover_every_chart_exactly_once_even_with_equal_timestamps(self, make_api):
        api = await make_api()
        headers, owner = await signup(api)
        made = [
            await add_chart(api, owner, title=f"s{i}", age_s=0 if i < 5 else i) for i in range(25)
        ]

        seen: list[str] = []
        cursor = None
        pages = 0
        while True:
            params = {"limit": 10, **({"cursor": cursor} if cursor else {})}
            page = (await api.client.get(LIST, headers=headers, params=params)).json()
            seen += [i["id"] for i in page["items"]]
            pages += 1
            cursor = page["next_cursor"]
            if cursor is None:
                break
        assert pages == 3
        assert len(seen) == len(set(seen)) == 25
        assert set(seen) == {str(c.id) for c in made}

    async def test_a_full_last_page_has_no_dangling_cursor(self, make_api):
        api = await make_api()
        headers, owner = await signup(api)
        for i in range(10):
            await add_chart(api, owner, age_s=i)
        page = (await api.client.get(LIST, headers=headers, params={"limit": 10})).json()
        assert len(page["items"]) == 10 and page["next_cursor"] is None

    async def test_other_users_charts_never_show_up(self, make_api):
        api = await make_api()
        ann, ann_id = await signup(api, "ann@example.com")
        _, bob_id = await signup(api, "bob@example.com")
        mine = await add_chart(api, ann_id, title="Mine")
        await add_chart(api, bob_id, title="Bob's")

        ids = [i["id"] for i in (await api.client.get(LIST, headers=ann)).json()["items"]]
        assert ids == [str(mine.id)]

    async def test_no_charts_is_an_empty_page(self, make_api):
        api = await make_api()
        headers, _ = await signup(api)
        assert (await api.client.get(LIST, headers=headers)).json() == {
            "items": [],
            "next_cursor": None,
        }

    @pytest.mark.parametrize("limit", [0, -1, 101, "abc"])
    async def test_limit_must_be_between_1_and_100(self, make_api, limit):
        api = await make_api()
        headers, _ = await signup(api)
        r = await api.client.get(LIST, headers=headers, params={"limit": limit})
        assert (r.status_code, r.json()["code"]) == (422, "validation_error")

    @pytest.mark.parametrize("cursor", ["garbage", "!!!", "YQ", ""])
    async def test_a_cursor_that_is_not_ours_is_a_validation_error(self, make_api, cursor):
        api = await make_api()
        headers, _ = await signup(api)
        r = await api.client.get(LIST, headers=headers, params={"cursor": cursor})
        assert (r.status_code, r.json()["code"]) == (422, "validation_error")
        assert r.json()["detail"][0]["field"] == "cursor"

    async def test_needs_a_signed_in_user(self, make_api):
        api = await make_api()
        assert (await api.client.get(LIST)).status_code == 401


class TestEtag:
    async def test_the_chart_carries_a_validator_and_is_not_cached_blindly(self, make_api):
        api = await make_api()
        headers, owner = await signup(api)
        chart = await add_chart(api, owner)
        r = await api.client.get(f"{LIST}/{chart.id}", headers=headers)
        assert r.status_code == 200
        assert r.headers["etag"].startswith('W/"1-')
        assert r.headers["cache-control"] == "private, no-cache"

    async def test_an_unchanged_chart_answers_304_without_a_body(self, make_api):
        api = await make_api()
        headers, owner = await signup(api)
        chart = await add_chart(api, owner)
        etag = (await api.client.get(f"{LIST}/{chart.id}", headers=headers)).headers["etag"]

        again = await api.client.get(
            f"{LIST}/{chart.id}", headers={**headers, "If-None-Match": etag}
        )
        assert again.status_code == 304 and again.content == b"" and again.headers["etag"] == etag

    @pytest.mark.parametrize("template", ["{e}", 'W/"other", {e}', "*"])
    async def test_every_form_of_if_none_match_that_matches_is_a_304(self, make_api, template):
        api = await make_api()
        headers, owner = await signup(api)
        chart = await add_chart(api, owner)
        etag = (await api.client.get(f"{LIST}/{chart.id}", headers=headers)).headers["etag"]
        sent = template.format(e=etag)
        r = await api.client.get(f"{LIST}/{chart.id}", headers={**headers, "If-None-Match": sent})
        assert r.status_code == 304

    async def test_a_new_version_gets_a_new_validator_and_the_full_body(self, make_api):
        api = await make_api()
        headers, owner = await signup(api)
        chart = await add_chart(api, owner)
        old = (await api.client.get(f"{LIST}/{chart.id}", headers=headers)).headers["etag"]

        await api.ingestion.charts.add(chart.model_copy(update={"version": 2}))
        r = await api.client.get(f"{LIST}/{chart.id}", headers={**headers, "If-None-Match": old})
        assert r.status_code == 200 and r.headers["etag"] != old and r.json()["version"] == 2

    async def test_a_mismatching_validator_still_returns_the_chart(self, make_api):
        api = await make_api()
        headers, owner = await signup(api)
        chart = await add_chart(api, owner)
        r = await api.client.get(
            f"{LIST}/{chart.id}", headers={**headers, "If-None-Match": 'W/"nope"'}
        )
        assert r.status_code == 200

    async def test_someone_elses_chart_is_still_404_even_with_a_matching_validator(self, make_api):
        api = await make_api()
        ann, ann_id = await signup(api, "ann@example.com")
        bob, _ = await signup(api, "bob@example.com")
        chart = await add_chart(api, ann_id)
        etag = (await api.client.get(f"{LIST}/{chart.id}", headers=ann)).headers["etag"]
        r = await api.client.get(f"{LIST}/{chart.id}", headers={**bob, "If-None-Match": etag})
        assert r.status_code == 404
