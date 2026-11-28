from __future__ import annotations
import re
from dataclasses import dataclass
from enum import Enum

from src.shared.domain.value_objects import ChordExtension, ChordQuality, RootNote

Q = ChordQuality
E = ChordExtension


class LabelStatus(str, Enum):
    OK       = "ok"
    PARTIAL  = "partial"
    NO_CHORD = "no_chord"
    UNMAPPED = "unmapped"


ROOT_IDS      = {r: i for i, r in enumerate(RootNote)}
QUALITY_IDS   = {q: i for i, q in enumerate(ChordQuality)}
EXTENSION_IDS = {e: i for i, e in enumerate(ChordExtension)}
BASS_NONE_ID  = 12

INTERVAL_SEMITONES: dict[str, int] = {
    "1": 0,
    "b2": 1, "2": 2, "#2": 3,
    "b3": 3, "3": 4, "#3": 5,
    "4": 5, "#4": 6,
    "b5": 6, "5": 7, "#5": 8,
    "b6": 8, "6": 9, "#6": 10,
    "b7": 10, "7": 11,
}

NOTE_TO_ID: dict[str, int] = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}


def parse_bass_id(label_raw: str | None, root_id: int | None) -> int:
    """Parse bass note class (0..11 = C..B, 12 = none / root position)."""
    if not label_raw or root_id is None or root_id < 0 or "/" not in label_raw:
        return BASS_NONE_ID

    _, _, bass_str = label_raw.strip().rpartition("/")
    if not bass_str or ")" in bass_str:
        return BASS_NONE_ID

    if bass_str in INTERVAL_SEMITONES:
        semitones = INTERVAL_SEMITONES[bass_str]
        if semitones == 0:
            return BASS_NONE_ID
        bass_id = (root_id + semitones) % 12
        return BASS_NONE_ID if bass_id == root_id else bass_id

    if bass_str in NOTE_TO_ID:
        bass_id = NOTE_TO_ID[bass_str]
        return BASS_NONE_ID if bass_id == root_id else bass_id

    return BASS_NONE_ID


@dataclass(frozen=True)
class MappedChord:

    status:    LabelStatus
    root:      RootNote | None = None
    quality:   ChordQuality | None = None
    extension: ChordExtension | None = None
    bass_id:   int = BASS_NONE_ID

    @property
    def root_id(self) -> int | None:
        return None if self.root is None else ROOT_IDS[self.root]

    @property
    def quality_id(self) -> int | None:
        return None if self.quality is None else QUALITY_IDS[self.quality]

    @property
    def ext_id(self) -> int | None:
        return None if self.extension is None else EXTENSION_IDS[self.extension]


_SHORTHAND: dict[str, tuple[ChordQuality | None, ChordExtension | None]] = {
    "":        (Q.major,    E.none),
    "maj":     (Q.major,    E.none),
    "min":     (Q.minor,    E.none),
    "m":       (Q.minor,    E.none),
    "dim":     (Q.dim,      E.none),
    "aug":     (Q.major,    E.sharp5),
    "sus":     (Q.sus4,     E.none),
    "sus2":    (Q.sus2,     E.none),
    "sus4":    (Q.sus4,     E.none),
    "7":       (Q.dominant, E.dom7),
    "maj7":    (Q.major,    E.maj7),
    "min7":    (Q.minor,    E.min7),
    "minmaj7": (Q.minor,    E.maj7),
    "dim7":    (Q.dim,      E.dom7),
    "hdim7":   (Q.dim,      E.min7),
    "aug7":    (Q.dominant, E.sharp5),
    "7sus4":   (Q.sus4,     E.dom7),
    "7sus2":   (Q.sus2,     E.dom7),
    "9sus4":   (Q.sus4,     E.nat9),
    "9":       (Q.dominant, E.nat9),
    "maj9":    (Q.major,    E.nat9),
    "min9":    (Q.minor,    E.nat9),
    "11":      (Q.sus4,     E.nat9),
    "maj11":   (Q.major,    E.nat11),
    "min11":   (Q.minor,    E.nat11),
    "13":      (Q.dominant, E.nat13),
    "maj13":   (Q.major,    E.nat13),
    "min13":   (Q.minor,    E.nat13),
    "6":       (Q.major,    None),
    "maj6":    (Q.major,    None),
    "min6":    (Q.minor,    None),
    "5":       (None,       None),
    "1":       (None,       None),
}

_ALTERATIONS: dict[str, ChordExtension] = {
    "b5":  E.flat5,
    "#5":  E.sharp5,
    "b9":  E.flat9,
    "#9":  E.sharp9,
    "#11": E.sharp11,
    "b13": E.flat13,
}

_NATURAL_RANK = {E.none: 0, E.maj7: 1, E.min7: 1, E.dom7: 1, E.sharp5: 0, E.flat5: 0,
                 E.add9: 2, E.nat9: 3, E.nat11: 4, E.nat13: 5}

_NATURAL_PITCH = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_NOTE_RE = re.compile(r"^([A-G])([#b]*)$")
_REST_RE = re.compile(r"^([^()]*)(?:\(([^()]*)\))?$")


def _parse_root(token: str) -> RootNote | None:
    """'Bb' → RootNote.As, 'Cb' → RootNote.B, 'E#' → RootNote.F. None if not a note."""
    m = _NOTE_RE.match(token.strip())
    if not m:
        return None
    idx = _NATURAL_PITCH[m.group(1)] + sum(1 if c == "#" else -1 for c in m.group(2))
    return RootNote.from_semitone(idx % 12)


def _strip_bass(label: str) -> str:
    """'C:maj/3' → 'C:maj'. The bass note is not part of the 3-head taxonomy."""
    if "/" in label:
        head, _, tail = label.rpartition("/")
        if ")" not in tail:
            return head
    return label


def _apply_degrees(
    quality: ChordQuality,
    ext: ChordExtension,
    degrees: list[str],
) -> tuple[ChordQuality, ChordExtension] | None:
    """Apply parenthesised Harte degrees, e.g. '7(b9,#11)'. None → cannot be mapped."""
    degrees = ["9" if d == "2" else d for d in degrees]
    alts     = [d for d in degrees if d in _ALTERATIONS]
    naturals = [d for d in degrees if d in ("9", "11", "13")]
    sevenths = [d for d in degrees if d in ("7", "b7")]
    ignored  = [d for d in degrees if d == "5"]
    unknown  = [d for d in degrees if d not in alts + naturals + sevenths + ignored]
    if unknown:
        return None

    # Missing 7th added through a degree
    if ext == E.none:
        for d in sevenths:
            if d == "7":
                ext = E.maj7
            else:
                ext = E.min7 if quality in (Q.minor, Q.dim) else E.dom7

    # min(b5) / min7(b5) = diminished triad / half-diminished
    if alts == ["b5"] and quality == Q.minor and ext in (E.none, E.min7):
        return Q.dim, ext

    if len(alts) >= 2:
        return quality, E.alt
    if len(alts) == 1:
        return quality, _ALTERATIONS[alts[0]]

    # Natural extensions: keep the highest layer
    current_rank = _NATURAL_RANK.get(ext, 0)
    for d in naturals:
        if d == "9":
            candidate = E.add9 if ext == E.none else E.nat9
        elif d == "11":
            candidate = E.nat11
        else:
            candidate = E.nat13
        if _NATURAL_RANK[candidate] > current_rank:
            ext, current_rank = candidate, _NATURAL_RANK[candidate]
    return quality, ext


_INTERVAL_CORE = {"1", "3", "b3", "5", "7", "b7", "bb7", "4", "2"}


def _from_interval_list(degrees: list[str]) -> tuple[ChordQuality, ChordExtension | None] | None:
    """Map a pure interval list, e.g. '(3,5,b7,13)' → (dominant, nat13). None → cannot be mapped.

    Used when the label has no shorthand (``F:(3,5,b7,b9)``): the quality comes from the third,
    the 7th decides dom7/maj7/min7/dim7, the remaining degrees are handled like parenthesised ones.

    Degree ``6``: with a 7th present it is the 13th (``F:(3,5,6,7,9)`` → nat13); without a 7th it
    is a 6th / 6-9 chord, which has no extension class → ``(quality, None)`` (PARTIAL, like 'maj6').
    """
    has = set(degrees)
    extras = [d for d in degrees if d not in _INTERVAL_CORE]

    if "b3" in has:
        quality = Q.dim if "b5" in has else Q.minor
        if quality == Q.dim:
            extras = [d for d in extras if d != "b5"]
    elif "3" in has:
        quality = Q.major
    elif has & {"4", "2"}:
        quality = Q.sus4 if "4" in has else Q.sus2          # both 2 and 4 present → sus4
    else:
        return None

    if "6" in has:
        if has & {"7", "b7", "bb7"}:
            extras = [d for d in extras if d != "6"] + ["13"]
        else:
            return quality, None

    ext = E.none
    if "7" in has:
        ext = E.maj7
    elif "bb7" in has:
        quality, ext = Q.dim, E.dom7                      # fully diminished (value_objects convention)
    elif "b7" in has:
        if quality == Q.major:
            quality, ext = Q.dominant, E.dom7
        else:
            ext = E.min7 if quality in (Q.minor, Q.dim) else E.dom7

    if not extras:
        return quality, ext
    return _apply_degrees(quality, ext, extras)


def map_harte(label: str | None) -> MappedChord:
    """Map one raw Harte / JAMS chord label to model class ids.

    Examples:
        'C:maj'      → C,  major,      none
        'Bb:7(b9)'   → A#, dominant,   b9
        'B:hdim7'    → B,  diminished, min7
        'C:maj6'     → C,  major,      None  (PARTIAL — no 6th class)
        'N'          → NO_CHORD
    """
    if label is None:
        return MappedChord(LabelStatus.UNMAPPED)

    raw = label.strip()
    if raw in ("", "N", "X", "N.C.", "NC"):
        return MappedChord(LabelStatus.NO_CHORD)

    root_token, _, rest = _strip_bass(raw).partition(":")
    root = _parse_root(root_token)
    if root is None:
        return MappedChord(LabelStatus.UNMAPPED)

    bass_id = parse_bass_id(raw, ROOT_IDS[root])

    m = _REST_RE.match(rest.strip())
    if m is None:
        return MappedChord(LabelStatus.UNMAPPED)
    shorthand = m.group(1).strip()
    degrees = [d.strip() for d in (m.group(2) or "").split(",")
               if d.strip() and not d.strip().startswith("*")]

    # Power chord / unison written as an interval list: 'F#:(1,5)' → only the root is reliable
    if shorthand == "" and degrees and all(d in ("1", "5") for d in degrees):
        return MappedChord(LabelStatus.PARTIAL, root, None, None, bass_id)

    # Interval-list label without shorthand: 'F:(3,5,b7,13)', 'F:(4,5,b7)' (sus), 'A:(1,2,4,5)'
    if shorthand == "" and any(d in ("3", "b3", "4", "2") for d in degrees):
        applied = _from_interval_list(degrees)
        if applied is None:
            return MappedChord(LabelStatus.UNMAPPED)
        status = LabelStatus.OK if applied[1] is not None else LabelStatus.PARTIAL
        return MappedChord(status, root, applied[0], applied[1], bass_id)

    if shorthand not in _SHORTHAND:
        return MappedChord(LabelStatus.UNMAPPED)
    quality, ext = _SHORTHAND[shorthand]

    if quality is not None and ext is not None and degrees:
        applied = _apply_degrees(quality, ext, degrees)
        if applied is None:
            return MappedChord(LabelStatus.UNMAPPED)
        quality, ext = applied

    status = LabelStatus.OK if (quality is not None and ext is not None) else LabelStatus.PARTIAL
    return MappedChord(status, root, quality, ext, bass_id)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset v4 / Model v6 Decomposed Taxonomy (14 Skeletons × 7 Tensions × 5 Inversions)
# ─────────────────────────────────────────────────────────────────────────────

SKELETON_NAMES_V6: tuple[str, ...] = (
    "MAJ_SHELL",   # 0: maj, add9, add4
    "MAJ6_SHELL",  # 1: 6, 6/9
    "MAJ7_SHELL",  # 2: maj7, maj9, maj7#11, maj9#11, maj13
    "DOM7_NAT",    # 3: 7, 9, 13, 9(13), 7#11, 9#11, 13#11
    "DOM7_ALT",    # 4: 7b5, 7#5, 7b9, 7#9, 7b13, 7alt, 13b9, 7(b5,b9), 7(#5,b9), ...
    "MIN_SHELL",   # 5: m, madd9, madd4
    "MIN6_SHELL",  # 6: m6, m6/9
    "MIN7_SHELL",  # 7: m7, m9, m11, m13
    "MIN_MAJ7",    # 8: m(maj7), m(maj9)
    "HDIM_SHELL",  # 9: m7b5, m11b5
    "DIM_SHELL",   # 10: dim, dim7
    "AUG_SHELL",   # 11: aug
    "SUS2_SHELL",  # 12: sus2, maj7sus2
    "SUS4_SHELL",  # 13: sus4, maj7sus4, 7sus4, 9sus4, 13sus4, 7b9sus4, 13b9sus4
)

TENSION_NAMES_7D: tuple[str, ...] = (
    "b9",      # bit 0 (1)
    "9",       # bit 1 (2)
    "#9",      # bit 2 (4)
    "4_11",    # bit 3 (8)
    "#11",     # bit 4 (16)
    "b13_#5",  # bit 5 (32)
    "6_13",    # bit 6 (64)
)
TENSION_BIT: dict[str, int] = {name: 1 << i for i, name in enumerate(TENSION_NAMES_7D)}

INVERSION_NAMES_5D: tuple[str, ...] = (
    "root_pos",     # 0: bass == root (or no slash)
    "inv1_3rd",     # 1: 1st inversion (3rd / sus degree in bass)
    "inv2_5th",     # 2: 2nd inversion (5th / b5 / #5 in bass)
    "inv3_7th",     # 3: 3rd inversion (7th / 6th in bass)
    "other_slash",  # 4: pedal / non-chord-tone slash bass (e.g. C/D, F/G)
)

_V6_NO_CHORD_TOKENS = frozenset({"", "N", "X", "N.C.", "NC"})


@dataclass(frozen=True)
class V6ChordTarget:
    """Decomposed chord supervision target for Dataset v4 / Model v6."""

    root_id: int          # 0..11 = C..B, 12 = N (no-chord), -1 = unmapped
    skel14_id: int        # 0..13 = SKELETON_NAMES_V6, -1 = N / missing_third / unmapped
    tension_mask: int     # uint8 bitmask (0..127) over TENSION_NAMES_7D
    tension_valid: bool   # True when chord color is supervised (False for N / power / unmapped)
    bass_id: int          # 0..11 = slash bass pitch class, 12 = none / root position
    inversion_id: int     # 0..4 = INVERSION_NAMES_5D, -1 = N / unmapped
    missing_third: bool   # True for power chords / dyads (:5, :(1,5), :1)


def compute_inversion_id(root_id: int, bass_id: int, skel14_id: int) -> int:
    """Classify `(root_id, bass_id, skel14_id)` into 5 harmonic inversion classes (`0..4`)."""
    if root_id < 0 or root_id >= 12:
        return -1
    if bass_id < 0 or bass_id >= 12 or bass_id == root_id:
        return 0  # root_pos

    iv = (bass_id - root_id) % 12
    # 1st inversion (3rd / sus tone)
    if skel14_id in (0, 1, 2, 3, 4, 11) and iv == 4:
        return 1
    if skel14_id in (5, 6, 7, 8, 9, 10) and iv == 3:
        return 1
    if skel14_id == 12 and iv == 2:
        return 1
    if skel14_id == 13 and iv == 5:
        return 1

    # 2nd inversion (5th, or b5 for dim/hdim, #5 for aug)
    if skel14_id in (9, 10) and iv == 6:
        return 2
    if skel14_id == 11 and iv == 8:
        return 2
    if skel14_id not in (9, 10, 11) and iv == 7:
        return 2
    if skel14_id == 4 and iv in (6, 8):
        return 2

    # 3rd inversion (7th or 6th)
    if skel14_id in (2, 8) and iv == 11:
        return 3
    if skel14_id in (3, 4, 7, 9, 13) and iv == 10:
        return 3
    if skel14_id in (1, 6, 10) and iv == 9:
        return 3

    return 4  # other_slash (pedal / passing non-chord bass)


def _v6_fallback_from_qe(
    root_id: int | None,
    quality_id: int | None,
    ext_id: int | None,
    bass_id: int = BASS_NONE_ID,
) -> V6ChordTarget:
    """Derive `V6ChordTarget` from legacy `(root_id, quality_id, ext_id, bass_id)` columns."""
    if root_id is None or root_id < 0:
        return V6ChordTarget(-1, -1, 0, False, BASS_NONE_ID, -1, False)
    if root_id == 12:
        return V6ChordTarget(12, -1, 0, False, BASS_NONE_ID, -1, False)

    b = bass_id if (0 <= bass_id < 12 and bass_id != root_id) else BASS_NONE_ID
    if quality_id is None or quality_id < 0 or quality_id > 5:
        return V6ChordTarget(root_id, -1, 0, False, b, compute_inversion_id(root_id, b, -1), True)

    q = int(quality_id)
    e = -1 if ext_id is None else int(ext_id)
    mask = 0
    valid = True

    if q == 0:  # major
        if e == 0:
            sk = 0
        elif e == 1:  # banned maj b5 -> clean MAJ_SHELL
            sk = 0
        elif e == 2:  # aug
            sk = 11
        elif e == 3:  # maj7
            sk = 2
        elif e == 6:  # add9
            sk, mask = 0, TENSION_BIT["9"]
        elif e == 8:  # maj9
            sk, mask = 2, TENSION_BIT["9"]
        elif e == 10:  # banned maj11 -> add4 (if no 7th) / maj9
            sk, mask = 0, TENSION_BIT["4_11"]
        elif e == 11:  # maj7#11
            sk, mask = 2, TENSION_BIT["#11"]
        elif e == 12:  # maj13
            sk, mask = 2, TENSION_BIT["9"] | TENSION_BIT["6_13"]
        else:
            sk, valid = 0, False
    elif q == 1:  # minor
        if e == 0:
            sk = 5
        elif e == 3:  # m(maj7)
            sk = 8
        elif e == 4:  # m7
            sk = 7
        elif e == 6:  # madd9
            sk, mask = 5, TENSION_BIT["9"]
        elif e in (7, 11, 13):  # banned m7b9, m7#11, m7b13 -> clean m7
            sk = 7
        elif e == 8:  # m9
            sk, mask = 7, TENSION_BIT["9"]
        elif e == 10:  # m11
            sk, mask = 7, TENSION_BIT["9"] | TENSION_BIT["4_11"]
        elif e == 12:  # m13
            sk, mask = 7, TENSION_BIT["9"] | TENSION_BIT["6_13"]
        else:
            sk, valid = 5, False
    elif q == 2:  # dominant
        if e == 5:
            sk = 3
        elif e == 8:
            sk, mask = 3, TENSION_BIT["9"]
        elif e == 10:  # banned dom11 -> 9sus4
            sk, mask = 13, TENSION_BIT["9"] | TENSION_BIT["4_11"]
        elif e == 11:
            sk, mask = 3, TENSION_BIT["#11"]
        elif e == 12:
            sk, mask = 3, TENSION_BIT["9"] | TENSION_BIT["6_13"]
        elif e == 1:  # 7b5
            sk, mask = 4, TENSION_BIT["#11"]
        elif e == 2:  # 7#5
            sk, mask = 4, TENSION_BIT["b13_#5"]
        elif e == 7:  # 7b9
            sk, mask = 4, TENSION_BIT["b9"]
        elif e == 9:  # 7#9
            sk, mask = 4, TENSION_BIT["#9"]
        elif e == 13:  # 7b13
            sk, mask = 4, TENSION_BIT["b13_#5"]
        elif e == 14:  # 7alt
            sk, mask = 4, TENSION_BIT["b9"] | TENSION_BIT["#9"] | TENSION_BIT["b13_#5"]
        else:
            sk, valid = 3, False
    elif q == 3:  # diminished
        sk = 9 if e == 4 else 10
        if e < 0:
            valid = False
    elif q == 4:  # sus2 (banned 7sus2 -> sus2)
        sk = 12
        if e < 0:
            valid = False
    else:  # sus4
        sk = 13
        if e == 8:
            mask = TENSION_BIT["9"] | TENSION_BIT["4_11"]
        elif e == 12:
            mask = TENSION_BIT["9"] | TENSION_BIT["4_11"] | TENSION_BIT["6_13"]
        elif e in (0, 3, 5):
            mask = TENSION_BIT["4_11"]
        elif e < 0:
            valid = False

    return V6ChordTarget(
        root_id=root_id,
        skel14_id=sk,
        tension_mask=mask,
        tension_valid=valid,
        bass_id=b,
        inversion_id=compute_inversion_id(root_id, b, sk),
        missing_third=False,
    )


def parse_v6_chord_targets(
    label_raw: str | None,
    root_id: int | None = None,
    quality_id: int | None = None,
    ext_id: int | None = None,
    bass_id: int | None = None,
) -> V6ChordTarget:
    """Parse a raw Harte chord label directly into v6 `(root_id, skel14_id, tension_mask, inversion_id)`.

    Rescues 6th-family (`:maj6`, `:min6`, `:maj6(9)`, `:(3,5,13)`, `:maj6(*5)`, `:min6(*b3)`),
    pop/jazz compound extensions (`add4`, `madd4`, `9sus4`, `13b9`, `7#9`, `7(#5,b9)`, `m11b5`),
    and eliminates all 7 banned chords (`maj11, dom11, m7b9, m7#11, m7b13, b5, 7sus2`).
    """
    if label_raw is None or (isinstance(label_raw, float) and label_raw != label_raw):
        return _v6_fallback_from_qe(
            root_id, quality_id, ext_id,
            BASS_NONE_ID if bass_id is None else int(bass_id),
        )

    raw = str(label_raw).strip()
    if raw in _V6_NO_CHORD_TOKENS:
        return V6ChordTarget(12, -1, 0, False, BASS_NONE_ID, -1, False)

    stripped = _strip_bass(raw)
    root_token, _, rest = stripped.partition(":")
    root_obj = _parse_root(root_token)
    if root_obj is None:
        return _v6_fallback_from_qe(
            root_id, quality_id, ext_id,
            BASS_NONE_ID if bass_id is None else int(bass_id),
        )
    rid = ROOT_IDS[root_obj]
    bid = parse_bass_id(raw, rid)
    if bid == BASS_NONE_ID and bass_id is not None and 0 <= int(bass_id) < 12 and int(bass_id) != rid:
        bid = int(bass_id)

    m = _REST_RE.match(rest.strip())
    if m is None:
        return _v6_fallback_from_qe(rid, quality_id, ext_id, bid)

    shorthand = m.group(1).strip()
    raw_degrees = [
        d.strip()
        for d in (m.group(2) or "").split(",")
        if d.strip() and not d.strip().startswith("*")
    ]
    deg_set = set(raw_degrees)

    # 1. Power chords / unisons (`:5`, `:1`, `:(1,5)`, `:5(b7)`) -> missing_third = True
    if shorthand in ("5", "1") or (shorthand == "" and raw_degrees and deg_set <= {"1", "5"}):
        return V6ChordTarget(rid, -1, 0, False, bid, compute_inversion_id(rid, bid, -1), True)

    # Helper to extract 7-D tension mask from extra degree tokens
    def _degrees_to_tensions(tokens: list[str], is_minor: bool = False) -> int:
        tm = 0
        for tok in tokens:
            if tok in ("b9", "b2"):
                if not is_minor:
                    tm |= TENSION_BIT["b9"]
            elif tok in ("9", "2"):
                tm |= TENSION_BIT["9"]
            elif tok in ("#9", "#2"):
                if not is_minor:
                    tm |= TENSION_BIT["#9"]
            elif tok in ("11", "4"):
                tm |= TENSION_BIT["4_11"]
            elif tok in ("#11", "#4", "b5"):
                if not is_minor:
                    tm |= TENSION_BIT["#11"]
            elif tok in ("b13", "#5", "b6"):
                if not is_minor:
                    tm |= TENSION_BIT["b13_#5"]
            elif tok in ("13", "6"):
                tm |= TENSION_BIT["6_13"]
        return tm

    # 2. Rescued 6th-family shorthands (`:maj6`, `:6`, `:min6`, `:maj6(9)`, `:min6(9)`, `:maj6(2)`)
    if shorthand in ("6", "maj6"):
        tm = TENSION_BIT["6_13"] | _degrees_to_tensions(raw_degrees, is_minor=False)
        return V6ChordTarget(rid, 1, tm, True, bid, compute_inversion_id(rid, bid, 1), False)

    if shorthand == "min6":
        tm = TENSION_BIT["6_13"] | _degrees_to_tensions(raw_degrees, is_minor=True)
        return V6ChordTarget(rid, 6, tm, True, bid, compute_inversion_id(rid, bid, 6), False)

    # 3. Pure interval lists (no shorthand), e.g. `:(3,5,13)`, `:(3,5,11)`, `:(b3,5,11)`, `:(3,5,b7,9,11)`
    if shorthand == "" and raw_degrees:
        has_3 = "3" in deg_set
        has_b3 = "b3" in deg_set
        has_7 = "7" in deg_set
        has_b7 = "b7" in deg_set
        has_bb7 = "bb7" in deg_set
        has_b5 = "b5" in deg_set
        has_s5 = "#5" in deg_set
        has_4 = ("4" in deg_set) or ("11" in deg_set)
        has_2 = ("2" in deg_set) or ("9" in deg_set)
        has_6 = ("6" in deg_set) or ("13" in deg_set)

        if has_b3:
            if has_b5:
                if has_bb7:
                    sk, tm = 10, 0
                elif has_b7:
                    sk = 9
                    tm = _degrees_to_tensions([d for d in raw_degrees if d != "b5"], is_minor=True)
                    if has_4:
                        tm |= TENSION_BIT["4_11"]
                else:
                    sk, tm = 10, 0
            elif has_7:
                sk = 8
                tm = _degrees_to_tensions(raw_degrees, is_minor=True)
            elif has_b7:
                sk = 7
                tm = _degrees_to_tensions(raw_degrees, is_minor=True)
            elif has_6:
                sk = 6
                tm = TENSION_BIT["6_13"] | _degrees_to_tensions(raw_degrees, is_minor=True)
            else:
                sk = 5
                tm = _degrees_to_tensions(raw_degrees, is_minor=True)
            return V6ChordTarget(rid, sk, tm, True, bid, compute_inversion_id(rid, bid, sk), False)

        if has_3:
            if has_b7:
                if has_4 and not ({"b9", "#9", "#11", "b5", "#5", "b13"} & deg_set):
                    # Pop/Jazz `:(3,5,b7,9,11)` -> `SUS4_SHELL` (9sus4 / 7sus4) to eliminate banned dom11
                    sk = 13
                    tm = TENSION_BIT["4_11"] | (TENSION_BIT["9"] if has_2 else 0) | (TENSION_BIT["6_13"] if has_6 else 0)
                else:
                    alt_hit = bool({"b9", "#9", "b5", "#5", "b13"} & deg_set)
                    sk = 4 if alt_hit else 3
                    tm = _degrees_to_tensions(raw_degrees, is_minor=False)
            elif has_7:
                sk = 2
                # Eliminate banned maj11 -> keep 9 / #11 / 13 only
                tm = _degrees_to_tensions([d for d in raw_degrees if d not in ("4", "11")], is_minor=False)
            elif has_6:
                # Rescued `:(3,5,13)` / `:(3,5,6)` / `:(3,5,6,9)` -> MAJ6_SHELL
                sk = 1
                tm = TENSION_BIT["6_13"] | _degrees_to_tensions(raw_degrees, is_minor=False)
            elif has_s5 and not has_b5:
                sk, tm = 11, 0
            else:
                # Includes `:(3,5,11)` -> MAJ_SHELL + 4_11 (`add4`), `:(3,5,9)` -> `add9`
                sk = 0
                tm = _degrees_to_tensions([d for d in raw_degrees if d not in ("b5", "#5", "b9", "#9", "b13")], is_minor=False)
            return V6ChordTarget(rid, sk, tm, True, bid, compute_inversion_id(rid, bid, sk), False)

        if has_4:
            sk = 13
            tm = TENSION_BIT["4_11"] | _degrees_to_tensions(raw_degrees, is_minor=False)
            return V6ChordTarget(rid, sk, tm, True, bid, compute_inversion_id(rid, bid, sk), False)
        if has_2:
            sk = 12
            tm = _degrees_to_tensions([d for d in raw_degrees if d not in ("2", "9", "b7")], is_minor=False)
            return V6ChordTarget(rid, sk, tm, True, bid, compute_inversion_id(rid, bid, sk), False)

    # 4. Standard shorthands + parenthesized degrees
    if shorthand in ("", "maj"):
        if "b7" in deg_set:
            alt_hit = bool({"b9", "#9", "b5", "#5", "b13"} & deg_set)
            sk = 4 if alt_hit else 3
            tm = _degrees_to_tensions(raw_degrees, is_minor=False)
        elif "7" in deg_set:
            sk = 2
            tm = _degrees_to_tensions([d for d in raw_degrees if d not in ("4", "11")], is_minor=False)
        elif ("6" in deg_set or "13" in deg_set) and not ({"7", "b7"} & deg_set):
            sk = 1
            tm = TENSION_BIT["6_13"] | _degrees_to_tensions(raw_degrees, is_minor=False)
        elif "#5" in deg_set and "b5" not in deg_set:
            sk, tm = 11, 0
        else:
            sk = 0
            # Eliminates banned `maj(b5)` while preserving `add9` and `add4`
            tm = _degrees_to_tensions([d for d in raw_degrees if d not in ("b5", "#5", "b9", "#9", "b13")], is_minor=False)
        return V6ChordTarget(rid, sk, tm, True, bid, compute_inversion_id(rid, bid, sk), False)

    if shorthand in ("min", "m"):
        if "b5" in deg_set:
            sk = 9 if "b7" in deg_set else 10
            tm = _degrees_to_tensions([d for d in raw_degrees if d != "b5"], is_minor=True)
        elif "7" in deg_set:
            sk = 8
            tm = _degrees_to_tensions(raw_degrees, is_minor=True)
        elif "b7" in deg_set:
            sk = 7
            tm = _degrees_to_tensions(raw_degrees, is_minor=True)
        elif ("6" in deg_set or "13" in deg_set):
            sk = 6
            tm = TENSION_BIT["6_13"] | _degrees_to_tensions(raw_degrees, is_minor=True)
        else:
            sk = 5
            tm = _degrees_to_tensions(raw_degrees, is_minor=True)
        return V6ChordTarget(rid, sk, tm, True, bid, compute_inversion_id(rid, bid, sk), False)

    if shorthand == "maj7":
        sk = 2
        tm = _degrees_to_tensions([d for d in raw_degrees if d not in ("4", "11")], is_minor=False)
        return V6ChordTarget(rid, sk, tm, True, bid, compute_inversion_id(rid, bid, sk), False)

    if shorthand == "maj9":
        sk = 2
        tm = TENSION_BIT["9"] | _degrees_to_tensions([d for d in raw_degrees if d not in ("4", "11")], is_minor=False)
        return V6ChordTarget(rid, sk, tm, True, bid, compute_inversion_id(rid, bid, sk), False)

    if shorthand == "maj11":
        # Banned maj11 -> maj9#11 if #11 present, else maj9
        sk = 2
        tm = TENSION_BIT["9"] | _degrees_to_tensions([d for d in raw_degrees if d not in ("4", "11")], is_minor=False)
        return V6ChordTarget(rid, sk, tm, True, bid, compute_inversion_id(rid, bid, sk), False)

    if shorthand == "maj13":
        sk = 2
        tm = TENSION_BIT["9"] | TENSION_BIT["6_13"] | _degrees_to_tensions([d for d in raw_degrees if d not in ("4", "11")], is_minor=False)
        return V6ChordTarget(rid, sk, tm, True, bid, compute_inversion_id(rid, bid, sk), False)

    if shorthand == "min7":
        if "b5" in deg_set:
            sk = 9
            tm = _degrees_to_tensions([d for d in raw_degrees if d != "b5"], is_minor=True)
        else:
            sk = 7
            tm = _degrees_to_tensions(raw_degrees, is_minor=True)
        return V6ChordTarget(rid, sk, tm, True, bid, compute_inversion_id(rid, bid, sk), False)

    if shorthand == "min9":
        sk = 7
        tm = TENSION_BIT["9"] | _degrees_to_tensions(raw_degrees, is_minor=True)
        return V6ChordTarget(rid, sk, tm, True, bid, compute_inversion_id(rid, bid, sk), False)

    if shorthand == "min11":
        sk = 9 if "b5" in deg_set else 7
        tm = TENSION_BIT["9"] | TENSION_BIT["4_11"] | _degrees_to_tensions([d for d in raw_degrees if d != "b5"], is_minor=True)
        return V6ChordTarget(rid, sk, tm, True, bid, compute_inversion_id(rid, bid, sk), False)

    if shorthand == "min13":
        sk = 7
        tm = TENSION_BIT["9"] | TENSION_BIT["6_13"] | _degrees_to_tensions(raw_degrees, is_minor=True)
        return V6ChordTarget(rid, sk, tm, True, bid, compute_inversion_id(rid, bid, sk), False)

    if shorthand == "minmaj7":
        sk = 8
        tm = _degrees_to_tensions(raw_degrees, is_minor=True)
        return V6ChordTarget(rid, sk, tm, True, bid, compute_inversion_id(rid, bid, sk), False)

    if shorthand == "hdim7":
        sk = 9
        tm = _degrees_to_tensions(raw_degrees, is_minor=True)
        return V6ChordTarget(rid, sk, tm, True, bid, compute_inversion_id(rid, bid, sk), False)

    if shorthand in ("dim", "dim7"):
        return V6ChordTarget(rid, 10, 0, True, bid, compute_inversion_id(rid, bid, 10), False)

    if shorthand == "aug":
        return V6ChordTarget(rid, 11, 0, True, bid, compute_inversion_id(rid, bid, 11), False)

    if shorthand == "aug7":
        tm = TENSION_BIT["b13_#5"] | _degrees_to_tensions(raw_degrees, is_minor=False)
        return V6ChordTarget(rid, 4, tm, True, bid, compute_inversion_id(rid, bid, 4), False)

    if shorthand in ("7", "9", "13"):
        base_tm = 0
        if shorthand == "9":
            base_tm = TENSION_BIT["9"]
        elif shorthand == "13":
            base_tm = TENSION_BIT["9"] | TENSION_BIT["6_13"]
        tm = base_tm | _degrees_to_tensions(raw_degrees, is_minor=False)
        alt_bits = TENSION_BIT["b9"] | TENSION_BIT["#9"] | TENSION_BIT["b13_#5"]
        has_b5 = "b5" in deg_set
        sk = 4 if ((tm & alt_bits) or has_b5) else 3
        return V6ChordTarget(rid, sk, tm, True, bid, compute_inversion_id(rid, bid, sk), False)

    if shorthand == "11":
        # Eliminate banned dom11 -> 9sus4
        tm = TENSION_BIT["9"] | TENSION_BIT["4_11"] | _degrees_to_tensions(raw_degrees, is_minor=False)
        return V6ChordTarget(rid, 13, tm, True, bid, compute_inversion_id(rid, bid, 13), False)

    if shorthand in ("sus2", "7sus2"):
        # Eliminate banned 7sus2 -> SUS2_SHELL
        return V6ChordTarget(rid, 12, 0, True, bid, compute_inversion_id(rid, bid, 12), False)

    if shorthand in ("sus", "sus4", "7sus4", "9sus4"):
        base_tm = TENSION_BIT["4_11"]
        if shorthand == "9sus4":
            base_tm |= TENSION_BIT["9"]
        tm = base_tm | _degrees_to_tensions(raw_degrees, is_minor=False)
        return V6ChordTarget(rid, 13, tm, True, bid, compute_inversion_id(rid, bid, 13), False)

    return _v6_fallback_from_qe(rid, quality_id, ext_id, bid)
