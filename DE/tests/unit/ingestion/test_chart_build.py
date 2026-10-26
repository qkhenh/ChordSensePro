from datetime import UTC, datetime
from uuid import uuid4

from src.analysis_contract.models import AnalysisResult, PlaybackSource, SourceKind
from src.ingestion.application.charts import build_chart
from src.ingestion.domain.job import new_job
from src.ingestion.infrastructure.mock_engine import FIXTURES_DIR

NOW = datetime(2026, 10, 26, 12, tzinfo=UTC)


def result() -> AnalysisResult:
    return AnalysisResult.model_validate_json(
        (FIXTURES_DIR / "ii_v_i_c.result.json").read_text(encoding="utf-8")
    )


def job(kind=SourceKind.FILE, url=None):
    return new_job(owner_id=uuid4(), source_kind=kind, title="Take Five", now=NOW, source_url=url)


def test_a_file_chart_plays_its_own_audio_through_the_api():
    j, r = job(), result()
    chart = build_chart(j, r, now=NOW)
    assert chart.owner_id == j.owner_id
    assert chart.title == "Take Five" and chart.artist is None
    assert chart.ai_events == r.chord_events and chart.metadata == r.metadata
    assert chart.overrides == () and not chart.is_customized and chart.forked_from is None
    assert chart.version == 1
    assert (chart.created_at, chart.updated_at) == (NOW, NOW)
    assert chart.source.kind == SourceKind.FILE and chart.source.url is None
    assert chart.source.content_hash == r.content_hash
    assert chart.source.playback == PlaybackSource(
        kind="file", audio_url=f"/api/v1/charts/{chart.id}/audio"
    )


def test_a_demo_chart_is_a_file_chart_with_demo_kind():
    chart = build_chart(job(SourceKind.DEMO), result(), now=NOW)
    assert chart.source.kind == SourceKind.DEMO
    assert chart.source.playback.kind == "file"


def test_a_link_chart_keeps_the_url_and_the_given_playback_and_artist():
    playback = PlaybackSource(kind="youtube", embed_id="dQw4w9WgXcQ")
    chart = build_chart(
        job(SourceKind.URL, "https://youtu.be/dQw4w9WgXcQ"),
        result(),
        playback=playback,
        artist="Rick",
        now=NOW,
    )
    assert chart.source.url == "https://youtu.be/dQw4w9WgXcQ"
    assert chart.source.playback == playback
    assert chart.artist == "Rick"


def test_every_chart_gets_its_own_id():
    assert build_chart(job(), result(), now=NOW).id != build_chart(job(), result(), now=NOW).id


def test_an_explicit_title_wins_over_the_job_title():
    assert build_chart(job(), result(), title="From the link", now=NOW).title == "From the link"
