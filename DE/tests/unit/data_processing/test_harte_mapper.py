"""Unit tests for HarteMapper — pure Python, no audio libs or database."""
import pytest

from src.data_processing.application.mappers.harte_mapper import (
    LabelStatus,
    map_harte,
)
from src.shared.domain.value_objects import ChordExtension as E
from src.shared.domain.value_objects import ChordQuality as Q
from src.shared.domain.value_objects import RootNote as R


def _triple(label: str):
    m = map_harte(label)
    return m.status, m.root, m.quality, m.extension


class TestBasicChords:
    def test_major(self):
        assert _triple("C:maj") == (LabelStatus.OK, R.C, Q.major, E.none)

    def test_minor_and_ids(self):
        m = map_harte("A:min")
        assert (m.root_id, m.quality_id, m.ext_id) == (9, 1, 0)

    def test_dim(self):
        assert _triple("B:dim") == (LabelStatus.OK, R.B, Q.dim, E.none)

    def test_bare_root_is_major(self):
        assert _triple("G") == (LabelStatus.OK, R.G, Q.major, E.none)

    def test_aug_is_major_sharp5(self):
        assert _triple("C:aug") == (LabelStatus.OK, R.C, Q.major, E.sharp5)

    def test_sus(self):
        assert _triple("C:sus4") == (LabelStatus.OK, R.C, Q.sus4, E.none)
        assert _triple("C:sus2") == (LabelStatus.OK, R.C, Q.sus2, E.none)

    def test_bare_sus_is_sus4(self):
        assert _triple("C:sus") == (LabelStatus.OK, R.C, Q.sus4, E.none)

    def test_sus2_and_sus4_have_distinct_ids(self):
        a, b = map_harte("C:sus2"), map_harte("C:sus4")
        assert a.quality_id != b.quality_id
        assert (a.quality_id, b.quality_id) == (4, 5)        # ids 0-3 unchanged, sus2=4, sus4=5

    @pytest.mark.parametrize("label, quality", [
        ("D#:(2,5)",   Q.sus2),      # Hooktheory style interval lists
        ("D:(4,5)",    Q.sus4),
        ("A:(2,4,5)",  Q.sus4),      # both present → sus4
    ])
    def test_sus_in_interval_form(self, label, quality):
        status, _, q, e = _triple(label)
        assert (status, q, e) == (LabelStatus.OK, quality, E.none)


class TestRootSpelling:
    def test_flat_becomes_sharp(self):
        assert map_harte("Bb:7").root == R.As

    def test_enharmonic_edge_cases(self):
        assert map_harte("Cb:maj").root == R.B
        assert map_harte("E#:min").root == R.F

    def test_kaggle_style_label(self):
        assert _triple("Eb:dim") == (LabelStatus.OK, R.Ds, Q.dim, E.none)


class TestSeventhsAndExtensions:
    @pytest.mark.parametrize("label,quality,ext", [
        ("C:7",       Q.dominant, E.dom7),
        ("C:maj7",    Q.major,    E.maj7),
        ("A:min7",    Q.minor,    E.min7),
        ("C:minmaj7", Q.minor,    E.maj7),
        ("B:hdim7",   Q.dim,      E.min7),
        ("B:dim7",    Q.dim,      E.dom7),
        ("C:9",       Q.dominant, E.nat9),
        ("C:maj9",    Q.major,    E.nat9),
        ("C:min9",    Q.minor,    E.nat9),
        ("C:min11",   Q.minor,    E.nat11),
        ("G:13",      Q.dominant, E.nat13),
        ("C:7sus4",   Q.sus4,     E.dom7),
        ("C:7sus2",   Q.sus2,     E.dom7),
    ])
    def test_shorthand(self, label, quality, ext):
        status, _, q, e = _triple(label)
        assert status == LabelStatus.OK
        assert (q, e) == (quality, ext)

    @pytest.mark.parametrize("label,ext", [
        ("G:7(b9)",     E.flat9),
        ("G:7(#9)",     E.sharp9),
        ("C:maj7(#11)", E.sharp11),
        ("G:7(b13)",    E.flat13),
        ("G:7(b5)",     E.flat5),
        ("G:7(b9,#11)", E.alt),          # two alterations → compound 'alt'
        ("C:maj(9)",    E.add9),         # 9th without 7th
        ("C:maj7(9)",   E.nat9),
        ("C:7(13)",     E.nat13),
    ])
    def test_degrees(self, label, ext):
        assert map_harte(label).extension == ext

    def test_min7_b5_is_half_diminished(self):
        assert _triple("B:min7(b5)") == (LabelStatus.OK, R.B, Q.dim, E.min7)

    def test_omitted_degree_is_ignored(self):
        assert _triple("C:maj(*3)") == (LabelStatus.OK, R.C, Q.major, E.none)


class TestBassAndStatus:
    def test_bass_note_is_stripped(self):
        assert _triple("F#:min/b3") == (LabelStatus.OK, R.Fs, Q.minor, E.none)
        assert _triple("C:maj7/5")[0] == LabelStatus.OK

    @pytest.mark.parametrize("label", ["N", "X", "", "  N "])
    def test_no_chord(self, label):
        assert map_harte(label).status == LabelStatus.NO_CHORD

    def test_sixth_chord_is_partial(self):
        m = map_harte("C:maj6")
        assert m.status == LabelStatus.PARTIAL
        assert (m.root_id, m.quality_id, m.ext_id) == (0, 0, None)

    def test_power_chord_keeps_only_root(self):
        m = map_harte("C:5")
        assert m.status == LabelStatus.PARTIAL
        assert (m.root_id, m.quality_id, m.ext_id) == (0, None, None)

    @pytest.mark.parametrize("label", ["C:xyz", "Am7", "H:maj", "C:7(b2)"])
    def test_garbage_is_unmapped(self, label):
        m = map_harte(label)
        assert m.status == LabelStatus.UNMAPPED
        assert m.root_id is None and m.quality_id is None and m.ext_id is None

    def test_none_is_unmapped(self):
        assert map_harte(None).status == LabelStatus.UNMAPPED


class TestIntervalListLabels:
    """JAAH writes altered dominants as bare interval lists, e.g. 'F:(3,5,b7,13)'."""

    @pytest.mark.parametrize("label, root, quality, ext", [
        ("F:(3,5,b7,13)",        R.F,  Q.dominant, E.nat13),
        ("F:(3,#5,b7)",          R.F,  Q.dominant, E.sharp5),
        ("G:(3,5,b7,b9)",        R.G,  Q.dominant, E.flat9),
        ("Ab:(3,b7,#11)",        R.Gs, Q.dominant, E.sharp11),
        ("Ab:(3,5,b7,9,#11)",    R.Gs, Q.dominant, E.sharp11),
        ("G:(3,b5,b7,b9)",       R.G,  Q.dominant, E.alt),
        ("C:(b3,5,b7)",          R.C,  Q.minor,    E.min7),
        ("C:(b3,b5,b7)",         R.C,  Q.dim,      E.min7),
        ("C:(b3,b5,bb7)",        R.C,  Q.dim,      E.dom7),
        ("C:(3,5,7)",            R.C,  Q.major,    E.maj7),
        ("C:(3,5)",              R.C,  Q.major,    E.none),
    ])
    def test_interval_lists(self, label, root, quality, ext):
        assert _triple(label) == (LabelStatus.OK, root, quality, ext)

    def test_bare_root_with_degrees_keeps_major_rule(self):
        # no explicit third -> legacy behaviour: 'C:(9)' = C major + 9 = add9
        assert _triple("C:(9)") == (LabelStatus.OK, R.C, Q.major, E.add9)


class TestSusAndPowerInIntervalForm:
    """Labels from ChoCo audio partitions and GuitarSet."""

    @pytest.mark.parametrize("label, root, quality, ext", [
        ("F:(4,5,b7)",            R.F, Q.sus4,  E.dom7),      # 7sus4 written as intervals
        ("A:(1,2,4,5)",           R.A, Q.sus4,  E.none),
        ("F:(4,5,b7,9,11,13)",    R.F, Q.sus4,  E.nat13),
        ("C:maj(2)",              R.C, Q.major, E.add9),      # added 2nd == add9
        ("G:maj(2)/2",            R.G, Q.major, E.add9),
        ("D#:sus2(7)/1",          R.Ds, Q.sus2, E.maj7),      # GuitarSet 'performed'
        ("A:min7(*5)",            R.A, Q.minor, E.min7),      # omitted fifth is ignored
    ])
    def test_sus_and_add9_forms(self, label, root, quality, ext):
        assert _triple(label) == (LabelStatus.OK, root, quality, ext)

    @pytest.mark.parametrize("label", ["F#:(1,5)", "C:(1)", "G:(1,5)"])
    def test_power_chord_interval_lists_are_partial_root_only(self, label):
        m = map_harte(label)
        assert m.status == LabelStatus.PARTIAL
        assert m.quality_id is None and m.ext_id is None and m.root_id is not None

    def test_maj6_with_omitted_fifth_is_still_partial(self):
        assert map_harte("G#:maj6(*5)/1").status == LabelStatus.PARTIAL


class TestSixthDegreeInIntervalForm:
    """JAAH writes 6/9 chords as interval lists: 'Db:(3,6,9)/5'."""

    @pytest.mark.parametrize("label, root, quality", [
        ("Db:(3,6,9)/5",   R.Cs, Q.major),     # 6/9 → no 6th class → quality only
        ("C:(3,5,6,9)",    R.C,  Q.major),
        ("A:(b3,5,6)",     R.A,  Q.minor),     # minor 6th
    ])
    def test_sixth_without_seventh_is_partial(self, label, root, quality):
        m = map_harte(label)
        assert m.status == LabelStatus.PARTIAL
        assert (m.root, m.quality, m.extension) == (root, quality, None)

    def test_sixth_with_seventh_is_the_13th(self):
        assert _triple("F:(3,5,6,7,9)") == (LabelStatus.OK, R.F, Q.major, E.nat13)
        assert _triple("F:(3,5,b7,6)") == (LabelStatus.OK, R.F, Q.dominant, E.nat13)


