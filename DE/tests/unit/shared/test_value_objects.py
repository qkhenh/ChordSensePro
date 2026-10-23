"""Unit tests for shared/domain — no database, no external deps required."""
import pytest
from datetime import datetime, timezone

from src.shared.domain.value_objects import (
    RootNote, ChordQuality, ChordExtension, ChordLabel, ChromaVector,
    VALID_COMBOS, valid_combo_mask,
)
from src.shared.domain.processing_result import ProcessingResult
from src.shared.domain.base_model import BaseEntity, AuditedEntity


class TestRootNote:
    def test_semitone_index_c_is_zero(self):
        assert RootNote.semitone_index(RootNote.C) == 0

    def test_semitone_index_b_is_eleven(self):
        assert RootNote.semitone_index(RootNote.B) == 11

    def test_from_semitone_wraps_octave(self):
        """B + 1 semitone should wrap to C."""
        b_idx = RootNote.semitone_index(RootNote.B)
        wrapped = RootNote.from_semitone((b_idx + 1) % 12)
        assert wrapped == RootNote.C


class TestChordLabel:
    def test_am7_display_name(self):
        label = ChordLabel(root=RootNote.A, quality=ChordQuality.minor, extension=ChordExtension.min7)
        assert label.display_name == "Am7"

    def test_cmaj_display_name(self):
        label = ChordLabel(root=RootNote.C, quality=ChordQuality.major, extension=ChordExtension.none)
        assert label.display_name == "C"

    def test_maj9_and_dominant9_are_distinct(self):
        """Regression: Cmaj9 and C9 both rendered as 'C9'."""
        maj9 = ChordLabel(RootNote.C, ChordQuality.major, ChordExtension.nat9)
        dom9 = ChordLabel(RootNote.C, ChordQuality.dominant, ChordExtension.nat9)
        assert maj9.display_name == "Cmaj9"
        assert dom9.display_name == "C9"

    @pytest.mark.parametrize("quality, ext, expected", [
        (ChordQuality.major, ChordExtension.nat11, "Cmaj11"),
        (ChordQuality.dominant, ChordExtension.nat11, "C11"),
        (ChordQuality.major, ChordExtension.nat13, "Cmaj13"),
        (ChordQuality.dominant, ChordExtension.nat13, "C13"),
        (ChordQuality.dominant, ChordExtension.dom7, "C7"),
        (ChordQuality.dominant, ChordExtension.sharp9, "C7#9"),
        (ChordQuality.minor, ChordExtension.min7, "Cm7"),
        (ChordQuality.minor, ChordExtension.maj7, "Cm(maj7)"),
        (ChordQuality.dim, ChordExtension.min7, "Cm7b5"),
        (ChordQuality.dim, ChordExtension.dom7, "Cdim7"),
        (ChordQuality.sus4, ChordExtension.dom7, "C7sus4"),
        (ChordQuality.sus2, ChordExtension.dom7, "C7sus2"),
        (ChordQuality.sus2, ChordExtension.none, "Csus2"),
        (ChordQuality.sus4, ChordExtension.none, "Csus4"),
        (ChordQuality.sus4, ChordExtension.nat9, "C9sus4"),
        (ChordQuality.major, ChordExtension.sharp5, "Caug"),
    ])
    def test_known_display_names(self, quality, ext, expected):
        assert ChordLabel(RootNote.C, quality, ext).display_name == expected

    def test_display_name_is_injective_over_valid_combos(self):
        """Every valid (quality, extension) pair must have its own unique name."""
        names = {}
        for q, e in VALID_COMBOS:
            name = ChordLabel(RootNote.C, q, e).display_name
            assert name not in names, f"{name!r} produced by both {names[name]} and {(q, e)}"
            names[name] = (q, e)
        assert len(names) == len(VALID_COMBOS)

    def test_invalid_combo_is_visible_not_aliased(self):
        """major + min7 is not a real chord: it must not look like any valid chord."""
        bad = ChordLabel(RootNote.C, ChordQuality.major, ChordExtension.min7)
        assert not bad.is_valid
        valid_names = {ChordLabel(RootNote.C, q, e).display_name for q, e in VALID_COMBOS}
        assert bad.display_name not in valid_names
        assert "[" in bad.display_name

    def test_pitch_shift_keeps_validity_and_changes_root_only(self):
        label = ChordLabel(RootNote.A, ChordQuality.dominant, ChordExtension.nat9)
        shifted = label.pitch_shift(3)
        assert shifted.display_name == "C9"
        assert shifted.is_valid == label.is_valid


class TestValidComboMask:
    def test_mask_shape_matches_class_counts(self):
        mask = valid_combo_mask()
        assert len(mask) == len(ChordQuality)
        assert all(len(row) == len(ChordExtension) for row in mask)

    def test_mask_agrees_with_valid_combos(self):
        mask = valid_combo_mask()
        qs, es = list(ChordQuality), list(ChordExtension)
        for qi, q in enumerate(qs):
            for ei, e in enumerate(es):
                assert mask[qi][ei] == ((q, e) in VALID_COMBOS)

    def test_every_quality_has_at_least_one_valid_extension(self):
        """Otherwise the cascaded head could be masked to nothing."""
        assert all(any(row) for row in valid_combo_mask())

    def test_known_impossible_pairs_are_masked(self):
        mask = valid_combo_mask()
        qs, es = list(ChordQuality), list(ChordExtension)
        assert not mask[qs.index(ChordQuality.major)][es.index(ChordExtension.min7)]
        assert not mask[qs.index(ChordQuality.minor)][es.index(ChordExtension.dom7)]
        assert not mask[qs.index(ChordQuality.dominant)][es.index(ChordExtension.none)]


class TestProcessingResult:
    def test_ok_unwrap(self):
        result = ProcessingResult.ok(42)
        assert result.unwrap() == 42

    def test_ok_is_ok(self):
        assert ProcessingResult.ok("x").is_ok is True
        assert ProcessingResult.ok("x").is_err is False

    def test_err_is_err(self):
        result = ProcessingResult.err("oops")
        assert result.is_err is True
        assert result.is_ok is False

    def test_unwrap_on_err_raises(self):
        with pytest.raises(AssertionError, match="unwrap"):
            ProcessingResult.err("fail").unwrap()

    def test_unwrap_or_returns_default_on_err(self):
        result = ProcessingResult.err("fail")
        assert result.unwrap_or(99) == 99

    def test_unwrap_or_returns_value_on_ok(self):
        result = ProcessingResult.ok(42)
        assert result.unwrap_or(99) == 42

    def test_ok_none_does_not_raise(self):
        """Edge case: Ok(None) should NOT raise on unwrap."""
        result = ProcessingResult.ok(None)
        assert result.is_ok is True
        assert result.unwrap() is None

    def test_repr_ok(self):
        assert "Ok(42)" in repr(ProcessingResult.ok(42))

    def test_repr_err(self):
        assert "Err" in repr(ProcessingResult.err("boom"))


class TestBaseEntity:
    def test_auto_uuid(self):
        e = BaseEntity()
        assert len(e.id) == 36   # UUID4 string

    def test_equality_by_id(self):
        e1 = BaseEntity(id="abc")
        e2 = BaseEntity(id="abc")
        assert e1 == e2

    def test_inequality_by_id(self):
        e1 = BaseEntity(id="abc")
        e2 = BaseEntity(id="xyz")
        assert e1 != e2

    def test_hash_by_id(self):
        e1 = BaseEntity(id="abc")
        e2 = BaseEntity(id="abc")
        assert hash(e1) == hash(e2)


class TestAuditedEntity:
    def test_touch_updates_updated_at(self):
        e = AuditedEntity()
        old = e.updated_at
        import time
        time.sleep(0.01)
        e.touch()
        assert e.updated_at > old
