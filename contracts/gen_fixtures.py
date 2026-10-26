"""Generate the shared contract fixtures in contracts/fixtures/.

Run from DE/ so the contract package and its dependencies resolve:

    rtk uv run python ../contracts/gen_fixtures.py

Fixtures are built through the Pydantic models, so they can never be invalid; the
tests in DE/tests/unit/analysis_contract/test_fixtures.py guard against drift.
Audio, chord names and ids are fictional - no real recordings are referenced.
"""

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "DE"))

from src.analysis_contract import (  # noqa: E402
    AnalysisMetadata,
    AnalysisRequest,
    AnalysisResult,
    BeatGrid,
    Chart,
    ChartSource,
    ChordEvent,
    ChordLabelDTO,
    Confidence,
    ErrorCode,
    EventOverride,
    JobError,
    JobProgress,
    JobStage,
    JobState,
    KeyEstimate,
    KeyHint,
    OverrideOp,
    PlaybackSource,
    SourceKind,
    TimeSignature,
    apply_overrides,
    chord_symbol,
)
from src.analysis_contract.models import AnalysisOptions  # noqa: E402
from src.shared.domain.value_objects import (  # noqa: E402
    ChordExtension as X,
)
from src.shared.domain.value_objects import (  # noqa: E402
    ChordQuality as Q,
)
from src.shared.domain.value_objects import (  # noqa: E402
    RootNote as R,
)

FIXTURES = ROOT / "fixtures"
NOW = datetime(2026, 10, 22, 9, 0, tzinfo=UTC)
OWNER = uuid5(NAMESPACE_URL, "chordsense/fixtures/demo-owner")
MODEL_VERSION = "fixture-model-0.1"
PIPELINE_VERSION = "fixture-pipeline-0.1"
YOUTUBE_FAKE_ID = "FIXTURE0001"  # deliberately not a real video id


def fid(name: str) -> UUID:
    return uuid5(NAMESPACE_URL, f"chordsense/fixtures/{name}")


def chash(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()


def ch(root, quality, ext=X.none, bass=None) -> ChordLabelDTO:
    return ChordLabelDTO(root=root, quality=quality, extension=ext, bass=bass)


# Reusable chords (sharps are canonical, like DE's RootNote).
Dm7 = ch(R.D, Q.minor, X.min7)
G7 = ch(R.G, Q.dominant, X.dom7)
Cmaj7 = ch(R.C, Q.major, X.maj7)
C = ch(R.C, Q.major)


def build_result(
    name,
    *,
    key,
    mode,
    bpm,
    sig,
    beats_per_bar,
    offset_ms,
    bars,
    downbeats=None,
) -> AnalysisResult:
    bar_ms = beats_per_bar * 60_000 / bpm
    if downbeats:
        assert len(downbeats) == len(bars)
        starts = list(downbeats)
        last_len = downbeats[-1] - downbeats[-2]
    else:
        starts = [offset_ms + round(i * bar_ms) for i in range(len(bars))]
        last_len = round(bar_ms)
    ends = [*starts[1:], starts[-1] + last_len]
    events = []
    for i, (label, start, end) in enumerate(zip(bars, starts, ends, strict=True)):
        events.append(
            ChordEvent(
                event_id=f"e{i + 1:04d}",
                start_ms=start,
                end_ms=end,
                chord=label,
                symbol=chord_symbol(label),
                confidence=Confidence(
                    root=round(0.96 - 0.02 * (i % 4), 2),
                    quality=round(0.92 - 0.03 * (i % 3), 2),
                    extension=round(0.88 - 0.04 * (i % 5), 2),
                ),
            )
        )
    return AnalysisResult(
        schema_version="1.0",
        job_id=fid(name),
        content_hash=chash(name),
        metadata=AnalysisMetadata(
            duration_ms=ends[-1] + 1000,
            key=KeyEstimate(tonic=key, mode=mode, confidence=0.9),
            beat_grid=BeatGrid(
                tempo_bpm=bpm,
                tempo_confidence=0.92,
                time_signature=sig,
                beats_per_bar=beats_per_bar,
                first_beat_offset_ms=starts[0] if downbeats else offset_ms,
                downbeats_ms=downbeats,
            ),
            model_version=MODEL_VERSION,
            pipeline_version=PIPELINE_VERSION,
        ),
        chord_events=events,
    )


TS_44 = TimeSignature(numerator=4, denominator=4)

RESULTS: dict[str, AnalysisResult] = {
    # Plain ii-V-I, zero offset: the simplest grid case.
    "ii_v_i_c": build_result(
        "ii_v_i_c", key=R.C, mode="major", bpm=120, sig=TS_44, beats_per_bar=4,
        offset_ms=0, bars=[Dm7, G7, Cmaj7, Cmaj7],
    ),
    # 12-bar blues with a non-zero first-beat offset (350 ms of intro).
    "blues_f": build_result(
        "blues_f", key=R.F, mode="major", bpm=100, sig=TS_44, beats_per_bar=4,
        offset_ms=350,
        bars=[ch(R.F, Q.dominant, X.dom7)] * 4
        + [ch(R.As, Q.dominant, X.dom7)] * 2
        + [ch(R.F, Q.dominant, X.dom7)] * 2
        + [ch(R.C, Q.dominant, X.dom7), ch(R.As, Q.dominant, X.dom7),
           ch(R.F, Q.dominant, X.dom7), ch(R.C, Q.dominant, X.dom7)],
    ),
    # Minor key, half-diminished and altered dominant, Bb/Eb spelled as A#/D# (sharp-canonical).
    "autumn_leaves_gm": build_result(
        "autumn_leaves_gm", key=R.G, mode="minor", bpm=110, sig=TS_44, beats_per_bar=4,
        offset_ms=480,
        bars=[
            ch(R.C, Q.minor, X.min7), ch(R.F, Q.dominant, X.dom7),
            ch(R.As, Q.major, X.maj7), ch(R.Ds, Q.major, X.maj7),
            ch(R.A, Q.dim, X.min7), ch(R.D, Q.dominant, X.flat9),
            ch(R.G, Q.minor), ch(R.G, Q.minor),
        ],
    ),
    # Slash chord and major-ninth (the Cmaj9 vs C9 distinction).
    "pop_slash_c": build_result(
        "pop_slash_c", key=R.C, mode="major", bpm=96, sig=TS_44, beats_per_bar=4,
        offset_ms=200,
        bars=[ch(R.C, Q.major, X.nat9), ch(R.G, Q.major, bass=R.B),
              ch(R.A, Q.minor, X.min7), ch(R.F, Q.major)] * 2,
    ),
    # 3/4
    "waltz_g_3_4": build_result(
        "waltz_g_3_4", key=R.G, mode="major", bpm=90,
        sig=TimeSignature(numerator=3, denominator=4), beats_per_bar=3, offset_ms=0,
        bars=[ch(R.G, Q.major), ch(R.E, Q.minor), ch(R.A, Q.minor, X.min7),
              ch(R.D, Q.dominant, X.dom7)] * 2,
    ),
    # Compound meter: 6/8 tracked in dotted quarters => 2 beats per bar.
    "compound_6_8_e": build_result(
        "compound_6_8_e", key=R.E, mode="minor", bpm=60,
        sig=TimeSignature(numerator=6, denominator=8), beats_per_bar=2, offset_ms=0,
        bars=[ch(R.E, Q.minor), ch(R.C, Q.major), ch(R.G, Q.major), ch(R.D, Q.major)],
    ),
    # Irregular tempo: explicit downbeats override anything interpolated from tempo.
    "rubato_downbeats_c": build_result(
        "rubato_downbeats_c", key=R.C, mode="major", bpm=100, sig=TS_44, beats_per_bar=4,
        offset_ms=0,
        bars=[C, ch(R.A, Q.minor, X.min7), Dm7, G7, C, Cmaj7],
        downbeats=[0, 2450, 4850, 7400, 9800, 12400],
    ),
}


def make_chart(slug, result, *, title, artist, source, overrides=(), forked_from=None, version=1):
    chart_id = fid(f"chart/{slug}")
    return Chart(
        id=chart_id,
        owner_id=OWNER,
        title=title,
        artist=artist,
        source=source,
        metadata=result.metadata,
        ai_events=result.chord_events,
        overrides=overrides,
        is_customized=bool(overrides),
        forked_from=forked_from,
        version=version,
        created_at=NOW,
        updated_at=NOW,
    )


def charts() -> dict[str, Chart]:
    ii = RESULTS["ii_v_i_c"]
    al = RESULTS["autumn_leaves_gm"]
    ii_id = fid("chart/ii_v_i_c")
    al_id = fid("chart/autumn_leaves_gm")
    al_events = {e.event_id: e for e in al.chord_events}
    edited_e3 = al_events["e0003"].model_copy(
        update={"chord": ch(R.As, Q.dominant, X.dom7), "symbol": "A#7", "is_user_edited": True}
    )
    return {
        # Uploaded/demo audio served by the gateway.
        "ii_v_i_c": make_chart(
            "ii_v_i_c", ii, title="ii-V-I in C (demo)", artist=None,
            source=ChartSource(
                kind=SourceKind.DEMO, content_hash=ii.content_hash,
                playback=PlaybackSource(kind="file", audio_url=f"/api/v1/charts/{ii_id}/audio"),
            ),
        ),
        # Link source: embedded YouTube player, the server keeps no audio.
        "autumn_leaves_gm": make_chart(
            "autumn_leaves_gm", al, title="Autumn Leaves (fixture)", artist="Public domain",
            source=ChartSource(
                kind=SourceKind.URL, url=f"https://www.youtube.com/watch?v={YOUTUBE_FAKE_ID}",
                content_hash=al.content_hash,
                playback=PlaybackSource(kind="youtube", embed_id=YOUTUBE_FAKE_ID),
            ),
        ),
        # User edits on top of the immutable AI baseline (set + delete).
        "autumn_leaves_gm.edited": make_chart(
            "autumn_leaves_gm.edited", al, title="Autumn Leaves (my changes)", artist="Public domain",
            source=ChartSource(
                kind=SourceKind.FILE, content_hash=al.content_hash,
                playback=PlaybackSource(
                    kind="file", audio_url=f"/api/v1/charts/{fid('chart/autumn_leaves_gm.edited')}/audio"
                ),
            ),
            overrides=[
                EventOverride(event_id="e0003", op=OverrideOp.SET, event=edited_e3),
                EventOverride(event_id="e0008", op=OverrideOp.DELETE),
            ],
            forked_from=al_id,
            version=3,
        ),
    }


def jobs():
    job = fid("ii_v_i_c")
    steps = [
        (JobState.QUEUED, None, 0.0),
        (JobState.PROCESSING, JobStage.NORMALIZING, 0.15),
        (JobState.PROCESSING, JobStage.SEPARATING, 0.45),
        (JobState.PROCESSING, JobStage.BEAT_TRACKING, 0.7),
        (JobState.PROCESSING, JobStage.RECOGNIZING, 0.9),
    ]
    sequence = [JobProgress(job_id=job, state=s, stage=st, progress=p) for s, st, p in steps]
    sequence.append(
        JobProgress(job_id=job, state=JobState.DONE, progress=1.0, result_chart_id=fid("chart/ii_v_i_c"))
    )
    messages = {
        ErrorCode.UNSUPPORTED_FORMAT: ("The file is not a supported or readable audio format.", False, JobStage.NORMALIZING),
        ErrorCode.FILE_TOO_LARGE: ("The file is larger than the 100 MB limit.", False, JobStage.NORMALIZING),
        ErrorCode.DURATION_TOO_LONG: ("The audio is longer than the 10 minute limit.", False, JobStage.NORMALIZING),
        ErrorCode.URL_UNRESOLVABLE: ("The link could not be resolved. Try uploading the file instead.", False, JobStage.FETCHING),
        ErrorCode.SSRF_BLOCKED: ("This address is not allowed.", False, JobStage.FETCHING),
        ErrorCode.QUOTA_EXCEEDED: ("Daily analysis quota used up.", False, None),
        ErrorCode.ANALYSIS_TIMEOUT: ("The analysis took too long and was cancelled.", True, JobStage.RECOGNIZING),
        ErrorCode.AI_FAILURE: ("The chord recognition service failed.", True, JobStage.RECOGNIZING),
    }
    errors = {}
    for code, (message, retryable, _stage) in messages.items():
        errors[code.value] = JobProgress(
            job_id=fid(f"error/{code.value}"),
            state=JobState.FAILED,
            stage=_stage,
            progress=0.35,
            error=JobError(code=code, message=message, retryable=retryable),
        )
    return sequence, errors


def requests():
    r = RESULTS["ii_v_i_c"]
    al = RESULTS["autumn_leaves_gm"]
    return {
        "file": AnalysisRequest(
            job_id=r.job_id, content_hash=r.content_hash, source_kind=SourceKind.FILE,
            audio_path="/data/tmp/fixtures/ii_v_i_c.wav",
        ),
        "url": AnalysisRequest(
            job_id=al.job_id, content_hash=al.content_hash, source_kind=SourceKind.URL,
            audio_path="/data/tmp/fixtures/autumn_leaves_gm.wav",
            source_url=f"https://www.youtube.com/watch?v={YOUTUBE_FAKE_ID}",
            options=AnalysisOptions(key_hint=KeyHint(tonic=R.G, mode="minor")),
        ),
    }


def write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(data, "model_dump"):
        data = data.model_dump(mode="json")
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    for name, result in RESULTS.items():
        write(FIXTURES / "results" / f"{name}.result.json", result)
    for name, chart in charts().items():
        write(FIXTURES / "charts" / f"{name}.chart.json", chart)
    edited = charts()["autumn_leaves_gm.edited"]
    effective = apply_overrides(edited.ai_events, edited.overrides)
    write(FIXTURES / "charts" / "autumn_leaves_gm.effective.json", [e.model_dump(mode="json") for e in effective])
    for name, req in requests().items():
        write(FIXTURES / "requests" / f"{name}.request.json", req)
    sequence, errors = jobs()
    write(FIXTURES / "jobs" / "success.sequence.json", [p.model_dump(mode="json") for p in sequence])
    write(FIXTURES / "jobs" / "errors.json", {k: v.model_dump(mode="json") for k, v in errors.items()})
    print(f"wrote fixtures to {FIXTURES}")  # noqa: T201


if __name__ == "__main__":
    main()
