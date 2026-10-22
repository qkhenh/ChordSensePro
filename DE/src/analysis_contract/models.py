"""Gateway <-> AI contract v1.0: the single source of truth for chart/job payloads.

Pydantic here generates ``contracts/openapi.json`` and, from it, the frontend types.
Time is always absolute integer milliseconds; the grid/bar view is derived by clients.
Roman numerals and scale suggestions are deliberately NOT part of this contract: they
are derived on the client by the Harmonic Engine (FR-04).
"""

import re
from enum import StrEnum
from itertools import pairwise
from typing import Annotated, Literal
from uuid import UUID

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from src.shared.domain.value_objects import (
    ChordExtension,
    ChordLabel,
    ChordQuality,
    RootNote,
)

SCHEMA_VERSION = "1.0"

Mode = Literal["major", "minor"]
UnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]
Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
EventId = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,32}$")]


class ContractModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --- enums ------------------------------------------------------------------------


class SourceKind(StrEnum):
    FILE = "file"
    URL = "url"
    RECORDING = "recording"
    DEMO = "demo"


class JobState(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class JobStage(StrEnum):
    FETCHING = "fetching"
    NORMALIZING = "normalizing"
    SEPARATING = "separating"
    BEAT_TRACKING = "beat_tracking"
    RECOGNIZING = "recognizing"


class ErrorCode(StrEnum):
    UNSUPPORTED_FORMAT = "unsupported_format"
    FILE_TOO_LARGE = "file_too_large"
    DURATION_TOO_LONG = "duration_too_long"
    URL_UNRESOLVABLE = "url_unresolvable"
    SSRF_BLOCKED = "ssrf_blocked"
    QUOTA_EXCEEDED = "quota_exceeded"
    ANALYSIS_TIMEOUT = "analysis_timeout"
    AI_FAILURE = "ai_failure"


ERROR_HTTP_STATUS: dict[ErrorCode, int] = {
    ErrorCode.UNSUPPORTED_FORMAT: 415,
    ErrorCode.FILE_TOO_LARGE: 413,
    ErrorCode.DURATION_TOO_LONG: 413,
    ErrorCode.URL_UNRESOLVABLE: 422,
    ErrorCode.SSRF_BLOCKED: 422,
    ErrorCode.QUOTA_EXCEEDED: 429,
    ErrorCode.ANALYSIS_TIMEOUT: 504,
    ErrorCode.AI_FAILURE: 502,
}


class OverrideOp(StrEnum):
    SET = "set"
    DELETE = "delete"
    INSERT = "insert"


# --- chord vocabulary -------------------------------------------------------------

# DE's ChordLabel.display_name renders Cmaj9 and C9 both as "C9" because the two share
# ChordExtension.nat9; for a MAJOR quality the natural 9/11/13 must read maj9/11/13.
_MAJOR_NATURAL_EXTENSIONS = {
    ChordExtension.nat9: "maj9",
    ChordExtension.nat11: "maj11",
    ChordExtension.nat13: "maj13",
}


class ChordLabelDTO(ContractModel):
    root: RootNote
    quality: ChordQuality
    extension: ChordExtension = ChordExtension.none
    bass: RootNote | None = None  # slash-chord bass; null when the model does not detect it

    @model_validator(mode="after")
    def _bass_differs_from_root(self):
        if self.bass is not None and self.bass == self.root:
            raise ValueError("bass must differ from root (omit it for root position)")
        return self


def chord_symbol(chord: ChordLabelDTO) -> str:
    """Canonical (Chordie-style) symbol: ``Am7``, ``Cmaj9``, ``G7/B``."""
    if chord.quality == ChordQuality.major and chord.extension in _MAJOR_NATURAL_EXTENSIONS:
        base = f"{chord.root.value}{_MAJOR_NATURAL_EXTENSIONS[chord.extension]}"
    else:
        base = ChordLabel(chord.root, chord.quality, chord.extension).display_name
    return f"{base}/{chord.bass.value}" if chord.bass is not None else base


class Confidence(ContractModel):
    root: UnitInterval
    quality: UnitInterval
    extension: UnitInterval


class ChordEvent(ContractModel):
    event_id: EventId
    start_ms: Annotated[int, Field(ge=0)]
    end_ms: int
    chord: ChordLabelDTO
    symbol: str
    confidence: Confidence | None = None
    is_user_edited: bool = False

    @model_validator(mode="after")
    def _consistent(self):
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        expected = chord_symbol(self.chord)
        if self.symbol != expected:
            raise ValueError(f"symbol {self.symbol!r} does not match chord (expected {expected!r})")
        return self


def validate_timeline(events, *, duration_ms: int | None = None) -> None:
    """Events must be sorted, unique by id, non-overlapping and inside the track."""
    ids = [e.event_id for e in events]
    if len(set(ids)) != len(ids):
        raise ValueError("event_id values must be unique")
    for prev, cur in pairwise(events):
        if cur.start_ms < prev.start_ms:
            raise ValueError("events must be sorted by start_ms")
        if cur.start_ms < prev.end_ms:
            raise ValueError(f"events {prev.event_id} and {cur.event_id} overlap")
    if duration_ms is not None and events and max(e.end_ms for e in events) > duration_ms:
        raise ValueError("an event ends after metadata.duration_ms")


# --- musical metadata -------------------------------------------------------------


class TimeSignature(ContractModel):
    numerator: Annotated[int, Field(ge=1, le=32)]
    denominator: Literal[2, 4, 8, 16]


def _strictly_increasing(values: list[int] | None) -> list[int] | None:
    if values is not None and any(b <= a for a, b in pairwise(values)):
        raise ValueError("downbeats_ms must be strictly increasing")
    return values


class BeatGrid(ContractModel):
    """``tempo_bpm`` counts the tracker's beats; ``beats_per_bar`` says how many per bar
    (6/8 tracked in dotted quarters has 2). ``downbeats_ms``, when present, overrides
    any grid interpolated from tempo (irregular tempo)."""

    tempo_bpm: Annotated[float, Field(gt=0, le=400)]
    tempo_confidence: UnitInterval
    time_signature: TimeSignature
    beats_per_bar: Annotated[int, Field(ge=1, le=12)]
    first_beat_offset_ms: Annotated[int, Field(ge=0)]
    downbeats_ms: Annotated[
        list[Annotated[int, Field(ge=0)]] | None, AfterValidator(_strictly_increasing)
    ] = None


class KeyEstimate(ContractModel):
    tonic: RootNote
    mode: Mode
    confidence: UnitInterval


class KeyHint(ContractModel):
    tonic: RootNote
    mode: Mode


class AnalysisMetadata(ContractModel):
    duration_ms: Annotated[int, Field(gt=0)]
    key: KeyEstimate
    beat_grid: BeatGrid
    model_version: Annotated[str, Field(min_length=1)]
    pipeline_version: Annotated[str, Field(min_length=1)]


# --- request / progress / result --------------------------------------------------


class AnalysisOptions(ContractModel):
    key_hint: KeyHint | None = None
    model_version: str | None = None


class AnalysisRequest(ContractModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    job_id: UUID
    content_hash: Sha256Hex
    source_kind: SourceKind
    audio_path: Annotated[
        str, Field(min_length=1)
    ]  # canonical WAV 44.1 kHz mono from the Normalizer
    source_url: str | None = None  # audit only; the AI never fetches
    options: AnalysisOptions = AnalysisOptions()

    @model_validator(mode="after")
    def _url_source(self):
        if self.source_kind == SourceKind.URL and not self.source_url:
            raise ValueError("source_url is required when source_kind is 'url'")
        if self.source_url is not None and not re.match(r"^https?://", self.source_url):
            raise ValueError("source_url must be an http(s) URL")
        return self


class JobError(ContractModel):
    code: ErrorCode
    message: Annotated[str, Field(min_length=1)]
    retryable: bool = False


class JobProgress(ContractModel):
    job_id: UUID
    state: JobState
    stage: JobStage | None = None
    progress: UnitInterval
    error: JobError | None = None
    result_chart_id: UUID | None = None

    @model_validator(mode="after")
    def _state_rules(self):
        if self.state == JobState.PROCESSING and self.stage is None:
            raise ValueError("stage is required while processing")
        if self.state == JobState.QUEUED and (self.stage is not None or self.progress != 0.0):
            raise ValueError("a queued job has no stage and zero progress")
        if self.state == JobState.DONE:
            if self.result_chart_id is None:
                raise ValueError("result_chart_id is required when done")
            if self.progress != 1.0:
                raise ValueError("progress must be 1.0 when done")
        if self.state == JobState.FAILED and self.error is None:
            raise ValueError("error is required when failed")
        if self.state != JobState.FAILED and self.error is not None:
            raise ValueError("error is only allowed when failed")
        if self.state != JobState.DONE and self.result_chart_id is not None:
            raise ValueError("result_chart_id is only allowed when done")
        return self


class AnalysisResult(ContractModel):
    schema_version: Literal["1.0"]
    job_id: UUID
    content_hash: Sha256Hex
    metadata: AnalysisMetadata
    chord_events: tuple[ChordEvent, ...]

    @model_validator(mode="after")
    def _timeline(self):
        validate_timeline(self.chord_events, duration_ms=self.metadata.duration_ms)
        return self


# --- stored chart (layered document) ----------------------------------------------


class PlaybackSource(ContractModel):
    """How the client plays the audio. YouTube/SoundCloud are embedded players: the
    server keeps no audio for them."""

    kind: Literal["file", "youtube", "soundcloud", "direct"]
    audio_url: str | None = None
    embed_id: str | None = None

    @model_validator(mode="after")
    def _shape(self):
        if self.kind in ("file", "direct"):
            if not self.audio_url or self.embed_id:
                raise ValueError(f"kind={self.kind} requires audio_url and forbids embed_id")
        elif not self.embed_id or self.audio_url:
            raise ValueError(f"kind={self.kind} requires embed_id and forbids audio_url")
        return self


class ChartSource(ContractModel):
    kind: SourceKind
    url: str | None = None
    content_hash: Sha256Hex
    playback: PlaybackSource


class EventOverride(ContractModel):
    event_id: EventId
    op: OverrideOp
    event: ChordEvent | None = None

    @model_validator(mode="after")
    def _shape(self):
        if self.op == OverrideOp.DELETE:
            if self.event is not None:
                raise ValueError("a delete override must not carry an event")
        elif self.event is None:
            raise ValueError(f"a {self.op.value} override requires an event")
        elif self.event.event_id != self.event_id:
            raise ValueError("override event_id must equal event.event_id")
        return self


class Chart(ContractModel):
    id: UUID
    owner_id: UUID
    title: Annotated[str, Field(min_length=1)]
    artist: str | None = None
    source: ChartSource
    metadata: AnalysisMetadata
    ai_events: tuple[ChordEvent, ...]  # immutable AI baseline
    overrides: tuple[EventOverride, ...] = ()  # personal layer
    is_customized: bool = False
    forked_from: UUID | None = None
    version: Annotated[int, Field(ge=1)]
    created_at: AwareDatetime
    updated_at: AwareDatetime

    @model_validator(mode="after")
    def _layers(self):
        validate_timeline(self.ai_events, duration_ms=self.metadata.duration_ms)
        baseline_ids = {e.event_id for e in self.ai_events}
        seen: set[str] = set()
        for o in self.overrides:
            if o.event_id in seen:
                raise ValueError(f"only one override per event is allowed ({o.event_id})")
            seen.add(o.event_id)
            if o.op == OverrideOp.INSERT and o.event_id in baseline_ids:
                raise ValueError(f"insert: event_id {o.event_id} already exists in the baseline")
            if o.op != OverrideOp.INSERT and o.event_id not in baseline_ids:
                raise ValueError(f"{o.op.value}: unknown baseline event {o.event_id}")
        return self


__all__ = [
    "ERROR_HTTP_STATUS",
    "SCHEMA_VERSION",
    "AnalysisMetadata",
    "AnalysisOptions",
    "AnalysisRequest",
    "AnalysisResult",
    "BeatGrid",
    "Chart",
    "ChartSource",
    "ChordEvent",
    "ChordLabelDTO",
    "Confidence",
    "ContractModel",
    "ErrorCode",
    "EventOverride",
    "JobError",
    "JobProgress",
    "JobStage",
    "JobState",
    "KeyEstimate",
    "KeyHint",
    "OverrideOp",
    "PlaybackSource",
    "SourceKind",
    "TimeSignature",
    "chord_symbol",
    "validate_timeline",
]
