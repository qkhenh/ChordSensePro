"""HarteMapper — raw Harte / JAMS chord label  →  (root, quality, extension) class ids.

The dataset annotations (JAAH, ChoCo/McGill JAMS, RWC, Kaggle filenames) all use
Harte-style strings such as ``"Bb:7(b9)"`` or ``"C:maj7"``. The AI model has three
classification heads (see 18_DE_AI_chordsense.md section 4.3):

    Head 1 — Root       12 classes  (RootNote order, sharps canonical)
    Head 2 — Quality     6 classes  (major, minor, dominant, diminished, sus2, sus4)
    Head 3 — Extension  15 classes  (ChordExtension order)

``map_harte`` converts one raw label into those ids and reports a ``LabelStatus``:

    ok        all three heads have a valid class
    partial   root known, but quality and/or extension has no class in the taxonomy
              (e.g. 6th chords, power chords) — trainer should mask the missing heads
    no_chord  'N' / 'X' (silence or no chord) — there is no 'N' class in the taxonomy
    unmapped  label could not be understood (kept in the DB so it can be re-mapped later
              without re-crawling)

Class ids are derived from enum definition order in ``shared.domain.value_objects``,
so they always stay in sync with the taxonomy documented in the spec.
"""
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


@dataclass(frozen=True)
class MappedChord:
    """Result of mapping one raw chord label."""
    status:    LabelStatus
    root:      RootNote | None = None
    quality:   ChordQuality | None = None
    extension: ChordExtension | None = None

    @property
    def root_id(self) -> int | None:
        return None if self.root is None else ROOT_IDS[self.root]

    @property
    def quality_id(self) -> int | None:
        return None if self.quality is None else QUALITY_IDS[self.quality]

    @property
    def ext_id(self) -> int | None:
        return None if self.extension is None else EXTENSION_IDS[self.extension]


# ── Shorthand table: Harte shorthand → (quality, extension) ──────────────────
# extension=None means "this chord type has no class in the 15-class taxonomy"
# (6th chords). Such labels become PARTIAL instead of being forced into a wrong class.
_SHORTHAND: dict[str, tuple[ChordQuality | None, ChordExtension | None]] = {
    "":        (Q.major,    E.none),
    "maj":     (Q.major,    E.none),
    "min":     (Q.minor,    E.none),
    "m":       (Q.minor,    E.none),
    "dim":     (Q.dim,      E.none),
    "aug":     (Q.major,    E.sharp5),     # aug absorbed into major + #5
    "sus":     (Q.sus4,     E.none),       # bare 'sus' conventionally means sus4
    "sus2":    (Q.sus2,     E.none),
    "sus4":    (Q.sus4,     E.none),
    "7":       (Q.dominant, E.dom7),
    "maj7":    (Q.major,    E.maj7),
    "min7":    (Q.minor,    E.min7),
    "minmaj7": (Q.minor,    E.maj7),
    "dim7":    (Q.dim,      E.dom7),       # fully diminished (value_objects convention)
    "hdim7":   (Q.dim,      E.min7),       # half diminished = m7b5
    "aug7":    (Q.dominant, E.sharp5),
    "7sus4":   (Q.sus4,     E.dom7),
    "7sus2":   (Q.sus2,     E.dom7),
    "9":       (Q.dominant, E.nat9),
    "maj9":    (Q.major,    E.nat9),
    "min9":    (Q.minor,    E.nat9),
    "11":      (Q.dominant, E.nat11),
    "maj11":   (Q.major,    E.nat11),
    "min11":   (Q.minor,    E.nat11),
    "13":      (Q.dominant, E.nat13),
    "maj13":   (Q.major,    E.nat13),
    "min13":   (Q.minor,    E.nat13),
    # 6th chords: root + quality are reliable, but there is no 6th extension class
    "6":       (Q.major,    None),
    "maj6":    (Q.major,    None),
    "min6":    (Q.minor,    None),
    # power chord / unison: only the root is reliable
    "5":       (None,       None),
    "1":       (None,       None),
}

# Single alterations that map straight to an extension class
_ALTERATIONS: dict[str, ChordExtension] = {
    "b5":  E.flat5,
    "#5":  E.sharp5,
    "b9":  E.flat9,
    "#9":  E.sharp9,
    "#11": E.sharp11,
    "b13": E.flat13,
}

# Natural extension degrees: higher layer wins when several are present
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
    degrees = ["9" if d == "2" else d for d in degrees]     # added 2nd == added 9th (maj(2) = add9)
    alts     = [d for d in degrees if d in _ALTERATIONS]
    naturals = [d for d in degrees if d in ("9", "11", "13")]
    sevenths = [d for d in degrees if d in ("7", "b7")]
    ignored  = [d for d in degrees if d == "5"]            # natural 5th — no information
    unknown  = [d for d in degrees if d not in alts + naturals + sevenths + ignored]
    if unknown:
        return None

    # Missing 7th added through a degree: 'sus4(b7)', 'maj(7)', ...
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
        return quality, E.alt                              # compound alteration
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

    m = _REST_RE.match(rest.strip())
    if m is None:
        return MappedChord(LabelStatus.UNMAPPED)
    shorthand = m.group(1).strip()
    degrees = [d.strip() for d in (m.group(2) or "").split(",")
               if d.strip() and not d.strip().startswith("*")]   # '*3' = omitted note

    # Power chord / unison written as an interval list: 'F#:(1,5)' → only the root is reliable
    if shorthand == "" and degrees and all(d in ("1", "5") for d in degrees):
        return MappedChord(LabelStatus.PARTIAL, root, None, None)

    # Interval-list label without shorthand: 'F:(3,5,b7,13)', 'F:(4,5,b7)' (sus), 'A:(1,2,4,5)'
    if shorthand == "" and any(d in ("3", "b3", "4", "2") for d in degrees):
        applied = _from_interval_list(degrees)
        if applied is None:
            return MappedChord(LabelStatus.UNMAPPED)
        status = LabelStatus.OK if applied[1] is not None else LabelStatus.PARTIAL
        return MappedChord(status, root, applied[0], applied[1])

    if shorthand not in _SHORTHAND:
        return MappedChord(LabelStatus.UNMAPPED)
    quality, ext = _SHORTHAND[shorthand]

    if quality is not None and ext is not None and degrees:
        applied = _apply_degrees(quality, ext, degrees)
        if applied is None:
            return MappedChord(LabelStatus.UNMAPPED)
        quality, ext = applied

    status = LabelStatus.OK if (quality is not None and ext is not None) else LabelStatus.PARTIAL
    return MappedChord(status, root, quality, ext)
