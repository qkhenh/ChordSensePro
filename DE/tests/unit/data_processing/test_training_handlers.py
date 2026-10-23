"""Tests for the new training-chain handlers: duration reader, AlignmentCheck, Label."""
import json
from pathlib import Path

import pytest

from src.data_processing.application.annotation_parsers.parsers import read_annotation_duration
from src.data_processing.application.pipeline_handlers.alignment_check_handler import AlignmentCheckHandler
from src.data_processing.application.pipeline_handlers.label_handler import LabelHandler
from src.data_processing.domain.models.audio_segment import AudioSegment
from src.data_processing.domain.models.processed_audio import ProcessedAudio


class _FakeAudioFile:
    """Duck-typed AudioFile — handlers only read these attributes."""
    def __init__(self, source_type: str):
        self.source_type = source_type
        self.source_url = "ytsearch1:x"
        self.analysis_id = "job-1"


def _pa(source: str, **kw) -> ProcessedAudio:
    return ProcessedAudio(audio_file=_FakeAudioFile(source), **kw)  # type: ignore[arg-type]


# ── read_annotation_duration ────────────────────────────────────────────────

def test_duration_from_jaah_json(tmp_path: Path):
    f = tmp_path / "song.json"
    f.write_text(json.dumps({"duration": 142.53, "parts": []}))
    assert read_annotation_duration(f) == pytest.approx(142.53)


def test_duration_from_jams_metadata(tmp_path: Path):
    f = tmp_path / "song.jams"
    f.write_text(json.dumps({"file_metadata": {"duration": 200.0}, "annotations": []}))
    assert read_annotation_duration(f) == pytest.approx(200.0)


def test_duration_none_for_lab_and_broken_files(tmp_path: Path):
    lab = tmp_path / "song.lab"
    lab.write_text("0 1 C:maj")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert read_annotation_duration(lab) is None
    assert read_annotation_duration(bad) is None
    assert read_annotation_duration(tmp_path / "missing.json") is None


# ── AlignmentCheckHandler ───────────────────────────────────────────────────

def test_alignment_skips_non_jaah():
    data = _pa("kaggle")
    out = AlignmentCheckHandler().handle(data)
    assert not out.is_failed and out.align_status == "n/a"


def test_alignment_skips_when_no_annotation():
    data = _pa("jaah", wav_path=Path("x.wav"))
    out = AlignmentCheckHandler().handle(data)
    assert not out.is_failed and out.align_status == "n/a"


def _wav(tmp_path: Path, seconds: float) -> Path:
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    path = tmp_path / "a.wav"
    sf.write(str(path), np.zeros(int(seconds * 8000), dtype="float32"), 8000)
    return path


def test_alignment_ok_within_tolerance(tmp_path: Path):
    wav = _wav(tmp_path, 10.0)
    ann = tmp_path / "ann.json"
    ann.write_text(json.dumps({"duration": 11.5, "parts": []}))
    out = AlignmentCheckHandler(tolerance_sec=3.0).handle(_pa("jaah", wav_path=wav, annotation_path=ann))
    assert not out.is_failed and out.align_status == "ok"


def test_alignment_rejects_mismatch(tmp_path: Path):
    wav = _wav(tmp_path, 10.0)
    ann = tmp_path / "ann.json"
    ann.write_text(json.dumps({"duration": 25.0, "parts": []}))
    out = AlignmentCheckHandler(tolerance_sec=3.0).handle(_pa("jaah", wav_path=wav, annotation_path=ann))
    assert out.is_failed and out.align_status == "duration_mismatch"
    assert "duration_mismatch" in (out.error or "")


# ── LabelHandler ────────────────────────────────────────────────────────────

def test_label_handler_fills_ids_and_status():
    segs = [
        AudioSegment(segment_idx=0, chord_label="C:maj7"),
        AudioSegment(segment_idx=1, chord_label="N"),
        AudioSegment(segment_idx=2, chord_label="C:6"),
        AudioSegment(segment_idx=3, chord_label=None),
    ]
    out = LabelHandler().handle(_pa("jaah", segments=segs))
    ok, no_chord, partial, unlabeled = out.segments
    assert ok.label_status == "ok" and ok.root_id == 0 and ok.ext_id is not None
    assert no_chord.label_status == "no_chord" and no_chord.root_id is None
    assert partial.label_status == "partial" and partial.root_id == 0
    assert unlabeled.label_status is None


def test_label_handler_short_circuits_on_error():
    data = _pa("jaah", segments=[AudioSegment(chord_label="C:maj")])
    data.error = "boom"
    out = LabelHandler().handle(data)
    assert out.segments[0].label_status is None
