"""Synthesises demo audio that matches a golden result's chord timeline and beat grid.

Pure standard library (``wave``, ``math``): a click on every beat and the chord's tones under
each event, so the player and the grid can be checked by ear and by instrument.
"""

import math
import wave
from array import array
from pathlib import Path

from src.analysis_contract.models import AnalysisResult, ChordLabelDTO, ChordQuality, RootNote
from src.ingestion.application.demo import DemoCatalog

SAMPLE_RATE = 44_100
_BASE = {
    ChordQuality.major: (0, 4, 7),
    ChordQuality.minor: (0, 3, 7),
    ChordQuality.dominant: (0, 4, 7),
    ChordQuality.dim: (0, 3, 6),
    ChordQuality.sus: (0, 5, 7),
}
_ADDED = {
    "maj7": (11,),
    "min7": (10,),
    "dom7": (10,),
    "add9": (2,),
    "nat9": (2,),
    "b9": (1,),
    "#9": (3,),
    "nat11": (5,),
    "#11": (6,),
    "nat13": (9,),
    "b13": (8,),
}
_FIFTH = {"b5": 6, "#5": 8}
_NOTE_GAIN = 0.12
_CLICK_GAIN = 0.35
_ACCENT_GAIN = 0.6
_CLICK_MS = 15
_EDGE_MS = 10


def chord_pitch_classes(chord: ChordLabelDTO) -> set[int]:
    """Pitch classes (0 = C) of the chord's tones; the bass does not change them."""
    intervals = list(_BASE[chord.quality])
    extension = chord.extension.value
    if extension in _FIFTH:
        intervals[2] = _FIFTH[extension]
    intervals.extend(_ADDED.get(extension, ()))
    root = RootNote.semitone_index(chord.root)
    return {(root + i) % 12 for i in intervals}


def _frames(ms: float) -> int:
    return round(ms * SAMPLE_RATE / 1000)


def _frequency(midi: int) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def _add_chord(buffer: list[float], chord: ChordLabelDTO, start: int, end: int) -> None:
    end = min(end, len(buffer))
    length = end - start
    if length <= 0:
        return
    edge = min(_frames(_EDGE_MS), length // 2) or 1
    tones = [_frequency(60 + (pc - 0) % 12) for pc in sorted(chord_pitch_classes(chord))]
    tones.append(_frequency(36 + RootNote.semitone_index(chord.root)))  # bass root
    for freq in tones:
        step = 2 * math.pi * freq / SAMPLE_RATE
        for i in range(length):
            gain = min(1.0, i / edge, (length - i) / edge)
            buffer[start + i] += _NOTE_GAIN * gain * math.sin(step * i)


def _add_click(buffer: list[float], at_ms: float, gain: float) -> None:
    start = _frames(at_ms)
    length = _frames(_CLICK_MS)
    step = 2 * math.pi * 1500 / SAMPLE_RATE
    for i in range(length):
        if start + i >= len(buffer):
            return
        buffer[start + i] += gain * math.exp(-5 * i / length) * math.sin(step * i)


def render(result: AnalysisResult) -> bytes:
    """16-bit mono PCM frames covering the whole song."""
    meta = result.metadata
    buffer = [0.0] * _frames(meta.duration_ms)
    for event in result.chord_events:
        _add_chord(buffer, event.chord, _frames(event.start_ms), _frames(event.end_ms))

    grid = meta.beat_grid
    accents = set(grid.downbeats_ms or ())
    beat_ms = 60_000 / grid.tempo_bpm
    k = 0
    while (at := grid.first_beat_offset_ms + k * beat_ms) < meta.duration_ms:
        is_down = round(at) in accents or (not accents and k % grid.beats_per_bar == 0)
        _add_click(buffer, at, _ACCENT_GAIN if is_down else _CLICK_GAIN)
        k += 1
    for at in accents:
        _add_click(buffer, at, _ACCENT_GAIN)

    peak = max(max(map(abs, buffer), default=0.0), 1.0)
    pcm = array("h", (int(32000 * sample / peak) for sample in buffer))
    return pcm.tobytes()


def generate_demo_audio(
    catalog: DemoCatalog, ids: list[str] | None = None, *, overwrite: bool = True
) -> list[Path]:
    """Write ``demo-<id>.wav`` for the given ids (default: all) and return the paths written.

    With ``overwrite=False`` a file that already exists is kept and not reported."""
    written = []
    for demo_id in ids or [t.id for t in catalog.tracks()]:
        result = catalog.result(demo_id)  # KeyError for an unknown id
        path = catalog.audio_path(demo_id)
        if not overwrite and path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(SAMPLE_RATE)
            out.writeframes(render(result))
        written.append(path)
    return written
