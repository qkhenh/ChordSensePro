"""Turning an analysis result into the stored, layered chart document."""

from datetime import datetime
from uuid import uuid4

from src.analysis_contract.models import AnalysisResult, Chart, ChartSource, PlaybackSource
from src.ingestion.domain.job import Job


def build_chart(
    job: Job,
    result: AnalysisResult,
    *,
    now: datetime,
    playback: PlaybackSource | None = None,
    artist: str | None = None,
    title: str | None = None,
) -> Chart:
    """``playback=None`` means the chart owns a WAV the API serves (file and demo charts)."""
    chart_id = uuid4()
    playback = playback or PlaybackSource(kind="file", audio_url=f"/api/v1/charts/{chart_id}/audio")
    return Chart(
        id=chart_id,
        owner_id=job.owner_id,
        title=title or job.title,
        artist=artist,
        source=ChartSource(
            kind=job.source_kind,
            url=job.source_url,
            content_hash=result.content_hash,
            playback=playback,
        ),
        metadata=result.metadata,
        ai_events=result.chord_events,
        version=1,
        created_at=now,
        updated_at=now,
    )
