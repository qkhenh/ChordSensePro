import json
import wave

import pytest

from src.analysis_contract.models import ChordLabelDTO, ChordQuality, RootNote
from src.ingestion.application.demo import DemoCatalog
from src.ingestion.application.demo_audio import (
    chord_pitch_classes,
    generate_demo_audio,
)
from src.ingestion.infrastructure.mock_engine import FIXTURES_DIR

IDS = [
    "ii_v_i_c",
    "blues_f",
    "autumn_leaves_gm",
    "pop_slash_c",
    "waltz_g_3_4",
    "compound_6_8_e",
    "rubato_downbeats_c",
]


def catalog(tmp_path) -> DemoCatalog:
    return DemoCatalog(tmp_path / "demo", FIXTURES_DIR)


def label(root, quality, extension="none", bass=None) -> ChordLabelDTO:
    return ChordLabelDTO(root=root, quality=quality, extension=extension, bass=bass)


class TestCatalog:
    def test_lists_the_seven_golden_songs_in_a_stable_order(self, tmp_path):
        assert [t.id for t in catalog(tmp_path).tracks()] == IDS

    def test_metadata_comes_from_the_fixtures(self, tmp_path):
        for track in catalog(tmp_path).tracks():
            fixture = json.loads((FIXTURES_DIR / f"{track.id}.result.json").read_text("utf-8"))
            assert track.duration_ms == fixture["metadata"]["duration_ms"]
            assert track.tempo_bpm == fixture["metadata"]["beat_grid"]["tempo_bpm"]
            assert track.title and track.artist

    def test_describes_the_key(self, tmp_path):
        keys = {t.id: t.key for t in catalog(tmp_path).tracks()}
        assert keys["ii_v_i_c"] == "C major"
        assert keys["autumn_leaves_gm"] == "G minor"

    @pytest.mark.parametrize("bad", ["nope", "../ii_v_i_c", "ii_v_i_c.wav", "", "II_V_I_C"])
    def test_only_catalogued_ids_resolve(self, tmp_path, bad):
        assert catalog(tmp_path).get(bad) is None

    def test_audio_lives_in_the_demo_directory_under_the_demo_stem(self, tmp_path):
        cat = catalog(tmp_path)
        assert cat.audio_path("ii_v_i_c") == tmp_path / "demo" / "demo-ii_v_i_c.wav"

    def test_the_golden_result_is_available(self, tmp_path):
        assert catalog(tmp_path).result("ii_v_i_c").metadata.duration_ms == 9000


class TestChordTones:
    def test_major_minor_and_sevenths(self):
        assert chord_pitch_classes(label(RootNote.C, ChordQuality.major)) == {0, 4, 7}
        assert chord_pitch_classes(label(RootNote.D, ChordQuality.minor, "min7")) == {2, 5, 9, 0}
        assert chord_pitch_classes(label(RootNote.G, ChordQuality.dominant, "dom7")) == {
            7,
            11,
            2,
            5,
        }
        assert chord_pitch_classes(label(RootNote.C, ChordQuality.major, "maj7")) == {0, 4, 7, 11}

    def test_half_diminished_and_suspended(self):
        assert chord_pitch_classes(label(RootNote.B, ChordQuality.dim, "min7")) == {
            11,
            2,
            5,
            9,
        }
        assert chord_pitch_classes(label(RootNote.C, ChordQuality.sus)) == {0, 5, 7}

    def test_altered_fifth_and_extensions(self):
        assert chord_pitch_classes(label(RootNote.C, ChordQuality.major, "b5")) == {0, 4, 6}
        assert chord_pitch_classes(label(RootNote.C, ChordQuality.major, "#5")) == {0, 4, 8}
        assert chord_pitch_classes(label(RootNote.C, ChordQuality.dominant, "#9")) == {0, 4, 7, 3}

    def test_transposes_with_the_root(self):
        assert chord_pitch_classes(label(RootNote.A, ChordQuality.minor)) == {9, 0, 4}

    def test_the_bass_note_does_not_change_the_chord_tones(self):
        slash = label(RootNote.C, ChordQuality.major, bass=RootNote.E)
        assert chord_pitch_classes(slash) == {0, 4, 7}


class TestGeneratedAudio:
    def test_writes_a_44k_mono_wav_as_long_as_the_song(self, tmp_path):
        cat = catalog(tmp_path)
        (path,) = generate_demo_audio(cat, ["ii_v_i_c"])
        assert path == cat.audio_path("ii_v_i_c")
        with wave.open(str(path)) as w:
            assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (44100, 1, 2)
            assert abs(w.getnframes() * 1000 / w.getframerate() - 9000) <= 10
            audible = w.readframes(w.getframerate())
        assert any(audible)  # not silence

    def test_the_same_song_always_renders_the_same_bytes(self, tmp_path):
        first = tmp_path / "a"
        second = tmp_path / "b"
        (one,) = generate_demo_audio(DemoCatalog(first, FIXTURES_DIR), ["compound_6_8_e"])
        (two,) = generate_demo_audio(DemoCatalog(second, FIXTURES_DIR), ["compound_6_8_e"])
        assert one.read_bytes() == two.read_bytes()

    def test_different_songs_sound_different(self, tmp_path):
        cat = catalog(tmp_path)
        one, two = generate_demo_audio(cat, ["ii_v_i_c", "compound_6_8_e"])
        assert one.read_bytes()[:20000] != two.read_bytes()[:20000]

    def test_unknown_ids_are_refused(self, tmp_path):
        with pytest.raises(KeyError):
            generate_demo_audio(catalog(tmp_path), ["nope"])


class TestKeepingExistingAudio:
    def test_without_overwrite_files_that_exist_are_left_alone(self, tmp_path):
        cat = catalog(tmp_path)
        (first,) = generate_demo_audio(cat, ["compound_6_8_e"])
        first.write_bytes(b"edited by hand")
        assert generate_demo_audio(cat, ["compound_6_8_e"], overwrite=False) == []
        assert first.read_bytes() == b"edited by hand"

    def test_without_overwrite_missing_files_are_still_written(self, tmp_path):
        cat = catalog(tmp_path)
        written = generate_demo_audio(cat, ["compound_6_8_e"], overwrite=False)
        assert written == [cat.audio_path("compound_6_8_e")] and written[0].exists()
