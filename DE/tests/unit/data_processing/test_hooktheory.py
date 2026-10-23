"""Tests for the Hooktheory (Sheet Sage) source: chord conversion, alignment, annotation builder,
parser, labelled-range gating, section length check and the same-video split rule."""
import json
from pathlib import Path

import pytest

from src.data_processing.application.annotation_parsers.hooktheory import (
    beat_to_time, build_annotation, harmony_label, usable_entry,
)
from src.data_processing.application.annotation_parsers.parsers import (
    HooktheoryParser, get_parser, read_annotation_duration, read_annotation_range,
    read_annotation_section,
)
from src.data_processing.application.mappers.harte_mapper import LabelStatus, map_harte
from src.data_processing.application.pipeline_handlers.alignment_check_handler import AlignmentCheckHandler
from src.data_processing.application.pipeline_handlers.annotation_handler import AnnotationHandler
from src.data_processing.domain.models.audio_segment import AudioSegment
from src.data_processing.domain.models.processed_audio import ProcessedAudio
from src.shared.domain.dataset_split import split_for


# ── harmony_label → map_harte ────────────────────────────────────────────────

@pytest.mark.parametrize("intervals, quality, ext", [
    ([4, 3],            "major",      "none"),
    ([3, 4],            "minor",      "none"),
    ([4, 3, 4],         "major",      "maj7"),
    ([3, 4, 3],         "minor",      "min7"),
    ([4, 3, 3],         "dominant",   "dom7"),
    ([3, 3],            "dim",        "none"),
    ([3, 3, 4],         "dim",        "min7"),       # half-diminished
    ([3, 3, 3],         "dim",        "dom7"),       # fully diminished (value_objects convention)
    ([5, 2],            "sus4",       "none"),
    ([2, 5],            "sus2",       "none"),
    ([5, 2, 3],         "sus4",       "dom7"),       # 7sus4
    ([4, 3, 7],         "major",      "add9"),
    ([3, 4, 7],         "minor",      "add9"),
    ([4, 3, 4, 3],      "major",      "nat9"),       # maj9
    ([3, 4, 3, 4],      "minor",      "nat9"),       # m9
    ([4, 3, 3, 4],      "dominant",   "nat9"),       # 9
    ([4, 3, 3, 4, 3],   "dominant",   "nat11"),
    ([4, 3, 3, 4, 3, 4], "dominant",  "nat13"),
    ([4, 4],            "major",      "sharp5"),     # augmented
])
def test_harmony_label_maps_to_expected_classes(intervals, quality, ext):
    mapped = map_harte(harmony_label(0, intervals))
    assert mapped.status == LabelStatus.OK, harmony_label(0, intervals)
    assert mapped.quality is not None and mapped.extension is not None
    assert (mapped.quality.name, mapped.extension.name) == (quality, ext)


def test_harmony_label_root_is_transposed():
    assert harmony_label(7, [4, 3]).startswith("G:")
    assert map_harte(harmony_label(7, [4, 3])).root_id == 7


@pytest.mark.parametrize("intervals, label", [([4, 3, 2], "C:maj6"), ([3, 4, 2], "C:min6")])
def test_added_sixth_uses_shorthand_and_is_partial(intervals, label):
    assert harmony_label(0, intervals) == label
    assert map_harte(label).status == LabelStatus.PARTIAL


def test_lone_root_and_power_chord_are_partial_root_only():
    assert map_harte(harmony_label(0, [])).status == LabelStatus.PARTIAL
    assert map_harte(harmony_label(0, [7])).status == LabelStatus.PARTIAL


# ── beat_to_time ─────────────────────────────────────────────────────────────

def test_beat_to_time_interpolates_and_extrapolates():
    beats, times = [0.0, 1.0, 2.0], [10.0, 10.5, 11.5]
    assert beat_to_time(0.5, beats, times) == pytest.approx(10.25)
    assert beat_to_time(1.5, beats, times) == pytest.approx(11.0)
    assert beat_to_time(3.0, beats, times) == pytest.approx(12.5)      # last interval = 1.0 s/beat
    assert beat_to_time(-1.0, beats, times) == pytest.approx(9.5)


# ── build_annotation ─────────────────────────────────────────────────────────

def _entry(**override):
    entry = {
        "tags": ["HARMONY", "AUDIO_AVAILABLE", "REFINED_ALIGNMENT", "NO_SWING"],
        "hooktheory": {"id": "abc123", "artist": "an-artist", "song": "a-song"},
        "youtube": {"id": "vid00000001", "duration": 200.0},
        "alignment": {"refined": {"beats": [float(i) for i in range(13)],
                                  "times": [50.0 + i * 0.5 for i in range(13)]}},   # 120 bpm from 50 s
        "annotations": {"harmony": [
            {"onset": 0, "offset": 4, "root_pitch_class": 0, "root_position_intervals": [4, 3], "inversion": 0},
            {"onset": 4, "offset": 8, "root_pitch_class": 7, "root_position_intervals": [5, 2], "inversion": 0},
            {"onset": 8, "offset": 12, "root_pitch_class": 9, "root_position_intervals": [3, 4, 7], "inversion": 0},
        ]},
    }
    entry.update(override)
    return entry


def test_build_annotation_clip_time_and_section():
    ann = build_annotation(_entry(), pad=3.0)
    assert ann is not None
    assert ann["format"] == "hooktheory" and ann["youtube_id"] == "vid00000001"
    assert ann["duration"] == 200.0
    assert ann["section"] == [47.0, 59.0]                       # labelled 50–56 s ± 3 s
    assert ann["labelled_range"] == [3.0, 9.0]
    assert [row[2] for row in ann["timeline"]] == ["C:(3,5)", "G:(4,5)", "A:(b3,5,9)"]
    assert ann["timeline"][0][:2] == [3.0, 5.0]                 # beat 0–4 = 50–52 s → clip 3–5 s


def test_build_annotation_clamps_section_to_video():
    entry = _entry(youtube={"id": "vid00000001", "duration": 57.0})
    ann = build_annotation(entry, pad=3.0)
    assert ann is not None and ann["section"][1] == 57.0


@pytest.mark.parametrize("mutation", [
    lambda e: e["tags"].remove("AUDIO_AVAILABLE"),
    lambda e: e["tags"].append("TEMPO_CHANGES"),
    lambda e: e["alignment"].pop("refined"),
    lambda e: e["annotations"].update(harmony=[]),
    lambda e: e["youtube"].update(duration=None),
])
def test_build_annotation_rejects_unusable_entries(mutation):
    entry = _entry()
    mutation(entry)
    assert build_annotation(entry) is None


def test_usable_entry_requires_youtube_id():
    assert usable_entry(_entry())
    assert not usable_entry(_entry(youtube={"id": "", "duration": 10.0}))


# ── parser / readers / sniffing ──────────────────────────────────────────────

@pytest.fixture
def ann_file(tmp_path: Path) -> Path:
    f = tmp_path / "abc123.json"
    f.write_text(json.dumps(build_annotation(_entry(), pad=3.0)), encoding="utf-8")
    return f


def test_get_parser_sniffs_hooktheory_json(ann_file: Path):
    assert isinstance(get_parser(ann_file), HooktheoryParser)


def test_parser_returns_clip_time_timeline(ann_file: Path):
    timeline = HooktheoryParser().parse(ann_file)
    assert timeline[0] == (3.0, 5.0, "C:(3,5)")
    assert len(timeline) == 3


def test_readers_for_hooktheory_and_other_formats(ann_file: Path, tmp_path: Path):
    assert read_annotation_section(ann_file) == (47.0, 59.0)
    assert read_annotation_range(ann_file) == (3.0, 9.0)
    assert read_annotation_duration(ann_file) == 200.0          # video duration → downloader version check

    jaah_like = tmp_path / "jaah.json"
    jaah_like.write_text(json.dumps({"duration": 100.0, "parts": []}))
    assert read_annotation_section(jaah_like) is None
    assert read_annotation_range(jaah_like) is None


# ── AnnotationHandler: only the labelled range is labelled ───────────────────

def _segments(starts):
    return [AudioSegment(song_id="s", segment_idx=i, start_ms=s * 1000.0, end_ms=(s + 2.0) * 1000.0)
            for i, s in enumerate(starts)]


def test_annotation_handler_leaves_padding_unlabeled(ann_file: Path):
    # labelled range is 3.0–9.0 s; segments are 2 s long (tolerance 0.25 s)
    data = ProcessedAudio(audio_file=None, annotation_path=ann_file,                   # type: ignore[arg-type]
                          segments=_segments([0.0, 1.5, 3.0, 4.5, 7.0, 8.5]))
    AnnotationHandler().handle(data)
    labels = [s.chord_label for s in data.segments]
    assert labels[0] is None and labels[1] is None          # starts before the clip (0–2, 1.5–3.5)
    assert labels[2] == "C:(3,5)"                           # 3–5 s, midpoint 4 s
    assert labels[3] == "G:(4,5)"                           # 4.5–6.5 s, midpoint 5.5 s
    assert labels[4] == "A:(b3,5,9)"                        # 7–9 s, midpoint 8 s
    assert labels[5] is None                                # ends at 10.5 s > 9.25


def test_annotation_handler_other_sources_keep_n_fallback(tmp_path: Path):
    lab = tmp_path / "song.lab"
    lab.write_text("0.0 2.0 C:maj\n")
    data = ProcessedAudio(audio_file=None, annotation_path=lab,                         # type: ignore[arg-type]
                          segments=_segments([0.0, 5.0]))
    AnnotationHandler().handle(data)
    assert [s.chord_label for s in data.segments] == ["C:maj", "N"]


# ── AlignmentCheckHandler: section length ────────────────────────────────────

class _FakeAudioFile:
    source_type = "hooktheory"
    source_url = "https://www.youtube.com/watch?v=vid00000001#ht=abc123"
    analysis_id = "job-1"


def test_alignment_check_compares_wav_with_section_length(ann_file: Path, tmp_path: Path):
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    wav = tmp_path / "clip.wav"
    sf.write(str(wav), np.zeros(44100 * 12, dtype="float32"), 44100)      # section = 12 s
    data = ProcessedAudio(audio_file=_FakeAudioFile(), wav_path=wav, annotation_path=ann_file)  # type: ignore[arg-type]
    assert AlignmentCheckHandler(tolerance_sec=3.0).handle(data).align_status == "ok"

    short = tmp_path / "short.wav"
    sf.write(str(short), np.zeros(44100 * 5, dtype="float32"), 44100)     # truncated download
    bad = ProcessedAudio(audio_file=_FakeAudioFile(), wav_path=short, annotation_path=ann_file)  # type: ignore[arg-type]
    result = AlignmentCheckHandler(tolerance_sec=3.0).handle(bad)
    assert result.align_status == "duration_mismatch" and result.is_failed


# ── split: all clips of a video share a split ────────────────────────────────

def test_split_ignores_hooktheory_clip_fragment():
    base = "https://www.youtube.com/watch?v=vid00000001"
    assert split_for(f"{base}#ht=aaa") == split_for(f"{base}#ht=bbb") == split_for(f"{base}#ht=ccc")


def test_split_unchanged_for_urls_without_fragment():
    assert split_for("ytsearch1:The Beatles Julia") == split_for("ytsearch1:The Beatles Julia")
    assert split_for("ytsearch1:F# something") in ("train", "val", "test")
