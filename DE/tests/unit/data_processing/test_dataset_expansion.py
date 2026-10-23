"""Tests for the multi-dataset expansion: new parsers, source routing, ChoCo search delegation,
alignment guard on ChoCo, and the seed scanners (GuitarSet / SWD / ChoCo release)."""
import importlib.util
import json
from pathlib import Path

import pytest

from src.data_processing.application.annotation_parsers.parsers import (
    JamsParser, RwcParser, SwdParser, get_parser,
)
from src.data_processing.application.pipeline_handlers.alignment_check_handler import AlignmentCheckHandler
from src.data_processing.application.pipeline_handlers.separate_handler import CLEAN_SOURCES
from src.data_processing.domain.models.processed_audio import ProcessedAudio

DE_ROOT = Path(__file__).resolve().parents[3]


def _chord_ann(values, data_source="expert_human"):
    return {
        "namespace": "chord",
        "annotation_metadata": {"data_source": data_source},
        "data": [{"time": i * 2.0, "duration": 2.0, "value": v, "confidence": None}
                 for i, v in enumerate(values)],
    }


# ── JamsParser: choose among several chord tracks ───────────────────────────

def test_jams_prefers_transcription_track(tmp_path: Path):
    """GuitarSet: instructed track first, 'performed' (note transcription) second → performed wins."""
    f = tmp_path / "gs.jams"
    f.write_text(json.dumps({"annotations": [
        _chord_ann(["D#:maj", "G#:maj"]),
        {"namespace": "pitch_contour", "annotation_metadata": {}, "data": {"0": []}},   # dict data: must not crash
        _chord_ann(["D#:sus2(7)/1", "G#:maj6(*5)/1"],
                   data_source="Semi-automatic chord transcription with manual verification"),
    ]}))
    timeline = JamsParser().parse(f)
    assert [t[2] for t in timeline] == ["D#:sus2(7)/1", "G#:maj6(*5)/1"]


def test_jams_keeps_first_track_when_no_transcription(tmp_path: Path):
    """Multi-annotator corpora (Chordify CASD) keep the old behaviour: first chord track."""
    f = tmp_path / "casd.jams"
    f.write_text(json.dumps({"annotations": [_chord_ann(["C:maj"]), _chord_ann(["A:min"])]}))
    assert [t[2] for t in JamsParser().parse(f)] == ["C:maj"]


def test_jams_without_chord_track_is_empty(tmp_path: Path):
    f = tmp_path / "none.jams"
    f.write_text(json.dumps({"annotations": [{"namespace": "beat", "data": []}]}))
    assert JamsParser().parse(f) == []


# ── SwdParser + get_parser CSV sniffing ─────────────────────────────────────

SWD_CSV = (
    'start;end;shorthand;extended;majmin;majmin_inv\n'
    '0.46;4.58;"C:min";"C:(b3,5)";"C:min";"C:min"\n'
    '4.58;5.2;"B:dim7/C";"B:(b3,b5,bb7)/C";"B:min";"B:min/C"\n'
    '9.3;9.92;\n'
)


def test_swd_parser_reads_shorthand_and_empty_as_no_chord(tmp_path: Path):
    f = tmp_path / "Schubert_D911-01_HU33.csv"
    f.write_text(SWD_CSV, encoding="utf-8")
    assert SwdParser().parse(f) == [
        (0.46, 4.58, "C:min"),
        (4.58, 5.2, "B:dim7/C"),
        (9.3, 9.92, "N"),
    ]


def test_get_parser_routes_swd_csv_but_keeps_rwc(tmp_path: Path):
    swd = tmp_path / "swd.csv"
    swd.write_text(SWD_CSV, encoding="utf-8")
    rwc = tmp_path / "rwc.csv"
    rwc.write_text("0.0\tC\n0.5\tAm\n", encoding="utf-8")
    assert isinstance(get_parser(swd), SwdParser)
    assert isinstance(get_parser(rwc), RwcParser)


# ── Source routing ──────────────────────────────────────────────────────────

def test_guitarset_is_clean_source_swd_is_not():
    assert "guitarset" in CLEAN_SOURCES      # solo guitar → no Demucs
    assert "swd" not in CLEAN_SOURCES        # voice + piano → separate


def test_new_sources_registered_in_dispatcher():
    from src.data_loader.application.entrypoints import _build_dispatcher
    from src.data_loader.application.downloaders.local_file_downloader import LocalFileDownloader
    from src.data_loader.domain.models.audio_file import AudioSource

    dispatcher = _build_dispatcher()
    for source in (AudioSource.GUITARSET, AudioSource.SWD):
        assert isinstance(dispatcher._registry[source], LocalFileDownloader)


# ── ChocoDownloader: ytsearch → JAAH search logic ───────────────────────────

def test_choco_ytsearch_delegates_to_jaah_downloader(monkeypatch):
    from src.data_loader.application.downloaders import jaah_downloader
    from src.data_loader.application.downloaders.choco_downloader import ChocoDownloader
    from src.data_loader.domain.models.audio_file import AudioFile, AudioSource
    from src.data_loader.domain.models.download_response import DownloadResponse

    calls = []

    def fake_download(self, audio_file):
        calls.append(audio_file.source_url)
        return DownloadResponse.err(audio_file, "JAAH download failed: boom")

    monkeypatch.setattr(jaah_downloader.JaahDownloader, "download", fake_download)
    audio = AudioFile(source_url="ytsearch1:The Beatles Julia", source_type=AudioSource.CHOCO,
                      expected_duration=174.0)
    response = ChocoDownloader().download(audio)

    assert calls == ["ytsearch1:The Beatles Julia"]
    assert response.is_err and response.error is not None and "ChoCo search download failed" in response.error


# ── AlignmentCheck now covers ChoCo ─────────────────────────────────────────

class _Audio:
    def __init__(self, source_type):
        self.source_type, self.source_url, self.analysis_id = source_type, "x", "job"


def test_alignment_enforced_for_choco(tmp_path: Path):
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    wav = tmp_path / "a.wav"
    sf.write(str(wav), np.zeros(8000 * 10, dtype="float32"), 8000)             # 10 s
    jams = tmp_path / "a.jams"
    jams.write_text(json.dumps({"file_metadata": {"duration": 100.0}, "annotations": []}))
    data = ProcessedAudio(audio_file=_Audio("choco"), wav_path=wav, annotation_path=jams)  # type: ignore[arg-type]
    out = AlignmentCheckHandler(tolerance_sec=3.0).handle(data)
    assert out.is_failed and out.align_status == "duration_mismatch"


# ── Seed scanners ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def seed():
    spec = importlib.util.spec_from_file_location("seed_local_datasets", DE_ROOT / "scripts" / "seed_local_datasets.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)      # psycopg2 is imported lazily, so this is safe locally
    return module


def test_scan_guitarset_pairs_audio_with_annotation(seed, tmp_path: Path):
    ann = tmp_path / "guitarset" / "annotation"
    audio = tmp_path / "guitarset" / "audio_mono-mic"
    ann.mkdir(parents=True), audio.mkdir(parents=True)
    (ann / "00_BN1-129-Eb_comp.jams").write_text("{}")
    (audio / "00_BN1-129-Eb_comp_mic.wav").write_bytes(b"")
    (ann / "orphan.jams").write_text("{}")                                    # no audio → skipped

    entries = seed.scan_guitarset(tmp_path)

    assert len(entries) == 1
    e = entries[0]
    assert e["source_type"] == "guitarset"
    assert e["source_url"] == "/data/datasets/guitarset/audio_mono-mic/00_BN1-129-Eb_comp_mic.wav"
    assert e["annotation_url"] == "/data/datasets/guitarset/annotation/00_BN1-129-Eb_comp.jams"


def test_scan_swd_only_versions_with_audio(seed, tmp_path: Path):
    root = tmp_path / "Schubert_Winterreise" / "Schubert_Winterreise_Dataset_v2-0"
    (root / "01_RawData" / "audio_wav").mkdir(parents=True)
    (root / "02_Annotations" / "ann_audio_chord").mkdir(parents=True)
    (root / "01_RawData" / "audio_wav" / "Schubert_D911-01_HU33.wav").write_bytes(b"")
    for version in ("HU33", "AL98"):                                          # AL98 has csv but no audio
        (root / "02_Annotations" / "ann_audio_chord" / f"Schubert_D911-01_{version}.csv").write_text("start;end")

    entries = seed.scan_swd(tmp_path)

    assert [e["source_type"] for e in entries] == ["swd"]
    assert entries[0]["annotation_url"].endswith("ann_audio_chord/Schubert_D911-01_HU33.csv")


def _release_jams(path: Path, *, title, performer, duration=200.0, kind="audio"):
    path.write_text(json.dumps({
        "file_metadata": {"title": title, "duration": duration},
        "sandbox": {"type": kind, "performers": [performer]},
        "annotations": [],
    }))


def test_scan_choco_release_builds_search_queries(seed, tmp_path: Path):
    jams = tmp_path / "choco_release" / "v1.0.0" / "v1.0.0" / "jams"
    jams.mkdir(parents=True)
    _release_jams(jams / "isophonics_0.jams", title="Julia", performer="The Beatles")
    _release_jams(jams / "isophonics_1.jams", title="Julia", performer="The Beatles")      # duplicate query
    _release_jams(jams / "isophonics_2.jams", title="Score only", performer="X", kind="score")
    _release_jams(jams / "jaah_0.jams", title="Evidence", performer="Thelonious Monk")     # excluded partition

    entries = seed.scan_choco_release(tmp_path)

    assert [e["source_url"] for e in entries] == ["ytsearch1:The Beatles Julia"]
    assert entries[0]["source_type"] == "choco"
    assert entries[0]["annotation_url"].endswith("jams/isophonics_0.jams")
