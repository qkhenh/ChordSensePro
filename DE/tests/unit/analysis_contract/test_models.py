"""Contract invariants for the Gateway <-> AI boundary (analysis_contract v1.0)."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from src.analysis_contract import (
    ERROR_HTTP_STATUS,
    AnalysisMetadata,
    AnalysisRequest,
    AnalysisResult,
    BeatGrid,
    ChordEvent,
    ChordLabelDTO,
    ErrorCode,
    JobError,
    JobProgress,
    JobStage,
    JobState,
    KeyEstimate,
    PlaybackSource,
    SourceKind,
    TimeSignature,
    chord_symbol,
)
from src.shared.domain.value_objects import ChordExtension, ChordQuality, RootNote

HASH = "a" * 64


def label(root=RootNote.C, quality=ChordQuality.major, ext=ChordExtension.none, bass=None):
    return ChordLabelDTO(root=root, quality=quality, extension=ext, bass=bass)


def event(event_id="e1", start=0, end=2000, **label_kwargs):
    chord = label(**label_kwargs)
    return ChordEvent(
        event_id=event_id,
        start_ms=start,
        end_ms=end,
        chord=chord,
        symbol=chord_symbol(chord),
    )


def metadata(duration_ms=10_000, **grid_kwargs):
    grid = {
        "tempo_bpm": 120.0,
        "tempo_confidence": 0.9,
        "time_signature": TimeSignature(numerator=4, denominator=4),
        "beats_per_bar": 4,
        "first_beat_offset_ms": 0,
    } | grid_kwargs
    return AnalysisMetadata(
        duration_ms=duration_ms,
        key=KeyEstimate(tonic=RootNote.C, mode="major", confidence=0.8),
        beat_grid=BeatGrid(**grid),
        model_version="m-0.1",
        pipeline_version="p-0.1",
    )


def result(events, **kwargs):
    return AnalysisResult(
        schema_version="1.0",
        job_id=uuid4(),
        content_hash=HASH,
        metadata=kwargs.pop("metadata", metadata()),
        chord_events=events,
        **kwargs,
    )


class TestChordSymbol:
    @pytest.mark.parametrize(
        ("root", "quality", "ext", "bass", "expected"),
        [
            (RootNote.C, ChordQuality.major, ChordExtension.none, None, "C"),
            (RootNote.A, ChordQuality.minor, ChordExtension.min7, None, "Am7"),
            (RootNote.G, ChordQuality.dominant, ChordExtension.dom7, None, "G7"),
            (RootNote.C, ChordQuality.major, ChordExtension.maj7, None, "Cmaj7"),
            (RootNote.B, ChordQuality.dim, ChordExtension.min7, None, "Bm7b5"),
            (RootNote.B, ChordQuality.dim, ChordExtension.dom7, None, "Bdim7"),
            (RootNote.C, ChordQuality.major, ChordExtension.sharp5, None, "Caug"),
            (RootNote.G, ChordQuality.dominant, ChordExtension.sharp9, None, "G7#9"),
            (RootNote.As, ChordQuality.major, ChordExtension.maj7, None, "A#maj7"),
            (RootNote.C, ChordQuality.major, ChordExtension.none, RootNote.E, "C/E"),
            (RootNote.G, ChordQuality.dominant, ChordExtension.dom7, RootNote.B, "G7/B"),
        ],
    )
    def test_known_symbols(self, root, quality, ext, bass, expected):
        assert chord_symbol(label(root, quality, ext, bass)) == expected

    def test_major_ninth_is_distinguished_from_dominant_ninth(self):
        # DE's ChordLabel.display_name renders both as "C9"; the contract must not.
        assert chord_symbol(label(quality=ChordQuality.major, ext=ChordExtension.nat9)) == "Cmaj9"
        assert chord_symbol(label(quality=ChordQuality.dominant, ext=ChordExtension.nat9)) == "C9"
        assert chord_symbol(label(quality=ChordQuality.minor, ext=ChordExtension.nat9)) == "Cm9"

    @pytest.mark.parametrize(
        ("ext", "expected"),
        [(ChordExtension.nat11, "Cmaj11"), (ChordExtension.nat13, "Cmaj13")],
    )
    def test_major_eleventh_and_thirteenth(self, ext, expected):
        assert chord_symbol(label(quality=ChordQuality.major, ext=ext)) == expected

    def test_bass_equal_to_root_is_rejected(self):
        with pytest.raises(ValidationError, match="bass"):
            label(root=RootNote.C, bass=RootNote.C)


class TestChordEvent:
    def test_end_must_be_after_start(self):
        with pytest.raises(ValidationError, match="end_ms"):
            event(start=1000, end=1000)

    def test_start_must_not_be_negative(self):
        with pytest.raises(ValidationError):
            event(start=-1, end=10)

    def test_symbol_must_match_structured_label(self):
        chord = label(root=RootNote.A, quality=ChordQuality.minor, ext=ChordExtension.min7)
        with pytest.raises(ValidationError, match="symbol"):
            ChordEvent(event_id="e1", start_ms=0, end_ms=10, chord=chord, symbol="Dm7")

    def test_confidence_is_bounded(self):
        chord = label()
        with pytest.raises(ValidationError):
            ChordEvent(
                event_id="e1",
                start_ms=0,
                end_ms=10,
                chord=chord,
                symbol="C",
                confidence={"root": 1.2, "quality": 0.5, "extension": 0.5},
            )

    def test_unknown_fields_are_rejected(self):
        chord = label()
        with pytest.raises(ValidationError):
            ChordEvent(event_id="e1", start_ms=0, end_ms=10, chord=chord, symbol="C", roman="I")

    def test_models_are_frozen(self):
        e = event()
        with pytest.raises(ValidationError):
            e.start_ms = 5

    def test_ai_contract_has_no_roman_numeral_field(self):
        # Roman numerals are derived on the client (FR-04); see memory.md ADR.
        assert "roman" not in ChordEvent.model_fields


class TestBeatGrid:
    def test_downbeats_must_be_strictly_increasing(self):
        with pytest.raises(ValidationError, match="downbeats_ms"):
            metadata(downbeats_ms=[0, 2000, 2000])

    def test_tempo_must_be_positive(self):
        with pytest.raises(ValidationError):
            metadata(tempo_bpm=0)

    def test_time_signature_denominator_is_a_power_of_two_note_value(self):
        with pytest.raises(ValidationError):
            TimeSignature(numerator=4, denominator=5)

    def test_compound_meter_counts_tracked_beats_not_numerator(self):
        grid = BeatGrid(
            tempo_bpm=60,
            tempo_confidence=0.7,
            time_signature=TimeSignature(numerator=6, denominator=8),
            beats_per_bar=2,
            first_beat_offset_ms=0,
        )
        assert grid.beats_per_bar == 2


class TestAnalysisResult:
    def test_valid_result(self):
        r = result([event("e1", 0, 2000), event("e2", 2000, 4000)])
        assert len(r.chord_events) == 2

    def test_events_must_not_overlap(self):
        with pytest.raises(ValidationError, match="overlap"):
            result([event("e1", 0, 2500), event("e2", 2000, 4000)])

    def test_events_must_be_sorted(self):
        with pytest.raises(ValidationError, match="sorted"):
            result([event("e2", 2000, 4000), event("e1", 0, 2000)])

    def test_event_ids_must_be_unique(self):
        with pytest.raises(ValidationError, match="unique"):
            result([event("e1", 0, 1000), event("e1", 1000, 2000)])

    def test_events_must_fit_inside_duration(self):
        with pytest.raises(ValidationError, match="duration"):
            result([event("e1", 0, 20_000)], metadata=metadata(duration_ms=10_000))

    def test_gaps_between_events_are_allowed(self):
        assert result([event("e1", 0, 1000), event("e2", 5000, 6000)])

    def test_empty_timeline_is_allowed(self):
        assert result([]).chord_events == ()

    def test_schema_version_is_pinned(self):
        with pytest.raises(ValidationError):
            AnalysisResult(
                schema_version="2.0",
                job_id=uuid4(),
                content_hash=HASH,
                metadata=metadata(),
                chord_events=[],
            )


class TestAnalysisRequest:
    def test_content_hash_must_be_sha256_hex(self):
        with pytest.raises(ValidationError):
            AnalysisRequest(
                job_id=uuid4(),
                content_hash="nothex",
                source_kind=SourceKind.FILE,
                audio_path="/data/work/a.wav",
            )

    def test_url_source_requires_source_url(self):
        with pytest.raises(ValidationError, match="source_url"):
            AnalysisRequest(
                job_id=uuid4(),
                content_hash=HASH,
                source_kind=SourceKind.URL,
                audio_path="/data/work/a.wav",
            )

    def test_url_must_be_http_or_https(self):
        with pytest.raises(ValidationError, match="source_url"):
            AnalysisRequest(
                job_id=uuid4(),
                content_hash=HASH,
                source_kind=SourceKind.URL,
                audio_path="/data/work/a.wav",
                source_url="file:///etc/passwd",
            )


class TestJobProgress:
    def test_queued_has_no_stage_and_zero_progress(self):
        p = JobProgress(job_id=uuid4(), state=JobState.QUEUED, progress=0.0)
        assert p.stage is None

    def test_processing_requires_a_stage(self):
        with pytest.raises(ValidationError, match="stage"):
            JobProgress(job_id=uuid4(), state=JobState.PROCESSING, progress=0.3)

    def test_done_requires_result_chart_and_full_progress(self):
        with pytest.raises(ValidationError, match="result_chart_id"):
            JobProgress(job_id=uuid4(), state=JobState.DONE, progress=1.0)
        with pytest.raises(ValidationError, match="progress"):
            JobProgress(job_id=uuid4(), state=JobState.DONE, progress=0.9, result_chart_id=uuid4())

    def test_failed_requires_error_and_forbids_result(self):
        err = JobError(code=ErrorCode.AI_FAILURE, message="boom")
        with pytest.raises(ValidationError, match="error"):
            JobProgress(job_id=uuid4(), state=JobState.FAILED, progress=0.4)
        with pytest.raises(ValidationError, match="result_chart_id"):
            JobProgress(
                job_id=uuid4(),
                state=JobState.FAILED,
                progress=0.4,
                error=err,
                result_chart_id=uuid4(),
            )

    def test_non_failed_states_forbid_error(self):
        with pytest.raises(ValidationError, match="error"):
            JobProgress(
                job_id=uuid4(),
                state=JobState.PROCESSING,
                stage=JobStage.SEPARATING,
                progress=0.2,
                error=JobError(code=ErrorCode.AI_FAILURE, message="x"),
            )

    def test_every_error_code_maps_to_an_http_status(self):
        assert set(ERROR_HTTP_STATUS) == set(ErrorCode)
        assert ERROR_HTTP_STATUS[ErrorCode.QUOTA_EXCEEDED] == 429
        assert ERROR_HTTP_STATUS[ErrorCode.FILE_TOO_LARGE] == 413
        assert ERROR_HTTP_STATUS[ErrorCode.UNSUPPORTED_FORMAT] == 415


class TestPlaybackSource:
    def test_file_requires_audio_url_only(self):
        assert PlaybackSource(kind="file", audio_url="/api/v1/charts/x/audio")
        with pytest.raises(ValidationError):
            PlaybackSource(kind="file")
        with pytest.raises(ValidationError):
            PlaybackSource(kind="file", audio_url="/a", embed_id="abc")

    def test_embedded_players_require_embed_id_only(self):
        assert PlaybackSource(kind="youtube", embed_id="abc123")
        with pytest.raises(ValidationError):
            PlaybackSource(kind="youtube")
        with pytest.raises(ValidationError):
            PlaybackSource(kind="soundcloud", embed_id="x", audio_url="/a")
