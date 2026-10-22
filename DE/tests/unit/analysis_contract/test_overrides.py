"""Layered chart semantics: the AI baseline is immutable, user edits are overrides."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from src.analysis_contract import (
    AnalysisMetadata,
    BeatGrid,
    Chart,
    ChartSource,
    ChordEvent,
    ChordLabelDTO,
    EventOverride,
    KeyEstimate,
    OverlapError,
    OverrideOp,
    PlaybackSource,
    SourceKind,
    TimeSignature,
    apply_overrides,
    chord_symbol,
)
from src.shared.domain.value_objects import ChordExtension, ChordQuality, RootNote

NOW = datetime(2026, 10, 22, tzinfo=UTC)


def ev(event_id, start, end, root=RootNote.C, quality=ChordQuality.major, ext=ChordExtension.none):
    chord = ChordLabelDTO(root=root, quality=quality, extension=ext)
    return ChordEvent(
        event_id=event_id,
        start_ms=start,
        end_ms=end,
        chord=chord,
        symbol=chord_symbol(chord),
    )


def chart(ai_events, overrides=()):
    return Chart(
        id=uuid4(),
        owner_id=uuid4(),
        title="t",
        source=ChartSource(
            kind=SourceKind.FILE,
            content_hash="b" * 64,
            playback=PlaybackSource(kind="file", audio_url="/api/v1/charts/x/audio"),
        ),
        metadata=AnalysisMetadata(
            duration_ms=10_000,
            key=KeyEstimate(tonic=RootNote.C, mode="major", confidence=0.9),
            beat_grid=BeatGrid(
                tempo_bpm=120,
                tempo_confidence=0.9,
                time_signature=TimeSignature(numerator=4, denominator=4),
                beats_per_bar=4,
                first_beat_offset_ms=0,
            ),
            model_version="m",
            pipeline_version="p",
        ),
        ai_events=ai_events,
        overrides=overrides,
        is_customized=bool(overrides),
        version=1,
        created_at=NOW,
        updated_at=NOW,
    )


AI = [ev("e1", 0, 2000), ev("e2", 2000, 4000, RootNote.D, ChordQuality.minor, ChordExtension.min7)]


class TestApplyOverrides:
    def test_no_overrides_returns_baseline_unchanged(self):
        assert apply_overrides(AI, []) == list(AI)

    def test_set_replaces_event_and_flags_it_user_edited(self):
        new = ev("e2", 2000, 4000, RootNote.G, ChordQuality.dominant, ChordExtension.dom7)
        out = apply_overrides(AI, [EventOverride(event_id="e2", op=OverrideOp.SET, event=new)])
        assert out[1].symbol == "G7"
        assert out[1].is_user_edited is True
        assert out[0].is_user_edited is False

    def test_delete_removes_event(self):
        out = apply_overrides(AI, [EventOverride(event_id="e1", op=OverrideOp.DELETE)])
        assert [e.event_id for e in out] == ["e2"]

    def test_insert_adds_event_in_time_order(self):
        extra = ev("u1", 4000, 6000)
        out = apply_overrides(AI, [EventOverride(event_id="u1", op=OverrideOp.INSERT, event=extra)])
        assert [e.event_id for e in out] == ["e1", "e2", "u1"]
        assert out[-1].is_user_edited is True

    def test_baseline_is_never_mutated(self):
        snapshot = list(AI)
        apply_overrides(AI, [EventOverride(event_id="e1", op=OverrideOp.DELETE)])
        assert list(AI) == snapshot
        assert all(e.is_user_edited is False for e in AI)

    def test_result_that_overlaps_is_rejected(self):
        bad = ev("e2", 1500, 4000)
        with pytest.raises(OverlapError):
            apply_overrides(AI, [EventOverride(event_id="e2", op=OverrideOp.SET, event=bad)])


class TestEventOverrideShape:
    def test_delete_must_not_carry_an_event(self):
        with pytest.raises(ValidationError, match="event"):
            EventOverride(event_id="e1", op=OverrideOp.DELETE, event=ev("e1", 0, 1000))

    def test_set_and_insert_require_an_event_with_matching_id(self):
        with pytest.raises(ValidationError, match="event"):
            EventOverride(event_id="e1", op=OverrideOp.SET)
        with pytest.raises(ValidationError, match="event_id"):
            EventOverride(event_id="e1", op=OverrideOp.SET, event=ev("zzz", 0, 1000))


class TestChartLayers:
    def test_set_and_delete_must_target_existing_baseline_events(self):
        with pytest.raises(ValidationError, match="unknown"):
            chart(AI, [EventOverride(event_id="nope", op=OverrideOp.DELETE)])

    def test_insert_must_use_a_fresh_event_id(self):
        with pytest.raises(ValidationError, match="already exists"):
            chart(AI, [EventOverride(event_id="e1", op=OverrideOp.INSERT, event=ev("e1", 0, 10))])

    def test_one_override_per_event(self):
        o = EventOverride(event_id="e1", op=OverrideOp.DELETE)
        with pytest.raises(ValidationError, match="one override"):
            chart(AI, [o, o])

    def test_version_starts_at_one(self):
        with pytest.raises(ValidationError):
            Chart.model_validate({**chart(AI).model_dump(mode="json"), "version": 0})
