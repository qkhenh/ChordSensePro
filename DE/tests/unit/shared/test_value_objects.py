"""Tests cho shared/domain/ — chạy không cần database, không cần audio libs."""
import pytest

# TODO: sau khi bạn viết value_objects.py, import vào đây và viết test


class TestRootNote:
    def test_semitone_index_c_is_zero(self):
        # from src.shared.domain.value_objects import RootNote
        # assert RootNote.semitone_index(RootNote.C) == 0
        pass

    def test_from_semitone_wraps_octave(self):
        # B + 1 semitone phải ra C (wrap 12→0)
        pass


class TestChordLabel:
    def test_am7_display_name(self):
        # ChordLabel(A, MINOR, MIN7).display_name == "Am7"
        pass

    def test_caug_is_major_plus_sharp5(self):
        # Caug = major + #5, KHÔNG phải quality="aug"
        pass

    def test_g7_sharp9(self):
        # ChordLabel(G, DOMINANT, SHARP9).display_name == "G7#9"
        pass

    def test_pitch_shift_wraps(self):
        # B.pitch_shift(1) → C
        pass


class TestProcessingResult:
    def test_ok_unwrap(self):
        # ProcessingResult.ok(42).unwrap() == 42
        pass

    def test_err_is_err(self):
        # ProcessingResult.err("oops").is_err == True
        pass

    def test_unwrap_on_err_raises(self):
        # ProcessingResult.err("x").unwrap() phải raise
        pass
