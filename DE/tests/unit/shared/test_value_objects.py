"""Unit tests for shared/domain — no database, no external deps required."""
import pytest
from datetime import datetime, timezone

from src.shared.domain.value_objects import RootNote, ChordQuality, ChordExtension, ChordLabel, ChromaVector
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
