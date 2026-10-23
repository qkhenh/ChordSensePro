"""Core music theory value objects: RootNote, ChordQuality, ChordExtension, ChromaVector, ChordLabel.

Naming convention follows Chordie.com. All objects are immutable and compared by value.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class RootNote(str, Enum):
    """12 chromatic pitch classes — sharps as canonical form.

    Enharmonic disambiguation (Db vs C#) is handled at the
    post-processing layer after key detection, not here.
    """
    C  = "C"
    Cs = "C#"
    D  = "D"
    Ds = "D#"
    E  = "E"
    F  = "F"
    Fs = "F#"
    G  = "G"
    Gs = "G#"
    A  = "A"
    As = "A#"
    B  = "B"

    @classmethod
    def semitone_index(cls, note: RootNote) -> int:
        """C=0, C#=1, D=2, ..., B=11"""
        return [n.value for n in cls].index(note.value)

    @classmethod
    def from_semitone(cls, idx: int) -> RootNote:
        """Used for pitch shift: (index + n_steps) % 12 → RootNote"""
        return list(cls)[idx % 12]


class ChordQuality(str, Enum):
    """6 chord qualities — 'aug' excluded, mapped to major + sharp5.

    Enum order == quality_id in the database / exported dataset (do not reorder).
    """
    major     = "major"
    minor     = "minor"
    dominant  = "dominant"   # implies b7, used for G7, G7#9, G7alt...
    dim       = "diminished" # 1-b3-b5
    sus2      = "sus2"       # 1-2-5
    sus4      = "sus4"       # 1-4-5 (a bare 'sus' label is read as sus4)


class ChordExtension(str, Enum):
    """15 extension classes — following Chordie.com naming convention."""
    # 5th alterations
    none   = "none"   # plain triad (C, Am, Bdim...)
    flat5  = "b5"     # e.g. Cm7b5, C7b5
    sharp5 = "#5"     # aug 5th — absorbs "aug" quality (e.g. Caug, C7#5)

    # 7th layer
    maj7   = "maj7"   # e.g. Cmaj7, Fmaj7
    min7   = "min7"   # e.g. Am7, Dm7
    dom7   = "dom7"   # e.g. G7, C7

    # 9th layer
    add9   = "add9"   # 9th WITHOUT 7th (e.g. Cadd9)
    flat9  = "b9"     # e.g. G7b9
    nat9   = "nat9"   # e.g. Cmaj9, Am9, G9
    sharp9 = "#9"     # Hendrix chord (e.g. G7#9)

    # 11th layer
    nat11  = "nat11"  # e.g. Fmaj11, Cm11
    sharp11 = "#11"   # Lydian sound (e.g. Cmaj7#11, G7#11)

    # 13th layer
    nat13  = "nat13"  # e.g. G13, Cmaj13
    flat13 = "b13"    # e.g. G7b13

    # compound alteration
    alt    = "alt"    # G7alt = b9+#9+#11+b13 combined


@dataclass(frozen=True)
class ChromaVector:
    """12-dim pitch class energy vector. Index 0=C, 1=C#, ..., 11=B."""
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.values) != 12:
            raise ValueError(f"ChromaVector requires 12 dims, got {len(self.values)}")
        if not all(0.0 <= v <= 1.0 for v in self.values):
            raise ValueError("ChromaVector values must be in [0.0, 1.0]")

    @classmethod
    def from_list(cls, values: Sequence[float]) -> ChromaVector:
        return cls(values=tuple(values))

    def dominant_pitch(self) -> RootNote:
        """Pitch class with highest energy — heuristic root detection."""
        idx = max(range(12), key=lambda i: self.values[i])
        return RootNote.from_semitone(idx)


@dataclass(frozen=True)
class ConfidenceScore:
    """Per-head confidence from 3 classification heads of the model."""
    root: float       # Head 1 [0, 1]
    quality: float    # Head 2 [0, 1]
    extension: float  # Head 3 [0, 1]

    def __post_init__(self) -> None:
        for name, val in [("root", self.root), ("quality", self.quality), ("extension", self.extension)]:
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"ConfidenceScore.{name} must be in [0.0, 1.0]")

    @property
    def overall(self) -> float:
        """Weighted overall — root is most important, extension hardest so lower weight.
        Tuneable after model training with a real validation set."""
        return 0.4 * self.root + 0.35 * self.quality + 0.25 * self.extension

    @property
    def is_reliable(self) -> bool:
        """True if confident enough to display to user.
        Threshold is empirical — will be tuned after model training."""
        return self.root >= 0.6 and self.quality >= 0.55


# ---------------------------------------------------------------------------
# Valid (quality, extension) combinations — single source of truth for:
#   1. display naming (suffix after the root), injective over valid combos
#   2. masking impossible head combinations at inference (see valid_combo_mask)
#
# The model has 3 independent heads, so it can output pairs that are not real chords
# (e.g. major + min7). Anything not listed here is invalid.
# NOTE: Csus2 (C-D-G) and Gsus4 (G-C-D) are the same three notes; telling them apart needs the bass /
# context, so expect more root confusion on suspended chords than on other qualities.
# ---------------------------------------------------------------------------
_Q = ChordQuality
_E = ChordExtension

CHORD_SUFFIX: dict[tuple[ChordQuality, ChordExtension], str] = {
    # --- major (maj7-family names for 7th+ because dominant owns the plain 7/9/11/13 names)
    (_Q.major, _E.none):    "",
    (_Q.major, _E.flat5):   "b5",
    (_Q.major, _E.sharp5):  "aug",        # Caug — 'aug' quality is folded into major + #5
    (_Q.major, _E.maj7):    "maj7",
    (_Q.major, _E.add9):    "add9",
    (_Q.major, _E.nat9):    "maj9",
    (_Q.major, _E.nat11):   "maj11",
    (_Q.major, _E.sharp11): "maj7#11",
    (_Q.major, _E.nat13):   "maj13",
    # --- minor
    (_Q.minor, _E.none):    "m",
    (_Q.minor, _E.maj7):    "m(maj7)",
    (_Q.minor, _E.min7):    "m7",
    (_Q.minor, _E.add9):    "madd9",
    (_Q.minor, _E.flat9):   "m7b9",
    (_Q.minor, _E.nat9):    "m9",
    (_Q.minor, _E.nat11):   "m11",
    (_Q.minor, _E.sharp11): "m7#11",
    (_Q.minor, _E.nat13):   "m13",
    (_Q.minor, _E.flat13):  "m7b13",
    # --- dominant (always implies a b7)
    (_Q.dominant, _E.flat5):   "7b5",
    (_Q.dominant, _E.sharp5):  "7#5",
    (_Q.dominant, _E.dom7):    "7",
    (_Q.dominant, _E.flat9):   "7b9",
    (_Q.dominant, _E.nat9):    "9",
    (_Q.dominant, _E.sharp9):  "7#9",
    (_Q.dominant, _E.nat11):   "11",
    (_Q.dominant, _E.sharp11): "7#11",
    (_Q.dominant, _E.nat13):   "13",
    (_Q.dominant, _E.flat13):  "7b13",
    (_Q.dominant, _E.alt):     "7alt",
    # --- diminished (Chordie convention: min7 = half-diminished, dom7 = fully diminished)
    (_Q.dim, _E.none):  "dim",
    (_Q.dim, _E.min7):  "m7b5",
    (_Q.dim, _E.dom7):  "dim7",
    # --- suspended 2nd (a 9th/11th/13th on top of a sus2 is not a meaningful chord)
    (_Q.sus2, _E.none):  "sus2",
    (_Q.sus2, _E.maj7):  "maj7sus2",
    (_Q.sus2, _E.dom7):  "7sus2",
    # --- suspended 4th
    (_Q.sus4, _E.none):  "sus4",
    (_Q.sus4, _E.maj7):  "maj7sus4",
    (_Q.sus4, _E.dom7):  "7sus4",
    (_Q.sus4, _E.nat9):  "9sus4",
    (_Q.sus4, _E.nat13): "13sus4",
}

VALID_COMBOS: frozenset[tuple[ChordQuality, ChordExtension]] = frozenset(CHORD_SUFFIX)


def valid_combo_mask() -> list[list[bool]]:
    """``mask[quality_id][ext_id]`` — True if the (quality, extension) pair is a real chord.

    Ids follow enum declaration order (same as ``training_samples.quality_id / ext_id``).
    Use at inference: after the quality head is decided, suppress extension logits whose mask
    entry is False before taking the argmax.
    """
    return [[(q, e) in VALID_COMBOS for e in ChordExtension] for q in ChordQuality]


@dataclass(frozen=True)
class ChordLabel:
    """Full chord label — final model output.

    Naming follows Chordie.com convention.
    Examples:
        ChordLabel(A, minor, min7)   → "Am7"
        ChordLabel(C, major, sharp5) → "Caug"
        ChordLabel(C, major, nat9)   → "Cmaj9"  (≠ dominant nat9 → "C9")
        ChordLabel(G, dominant, sharp9) → "G7#9"
        ChordLabel(B, dim, min7)     → "Bm7b5"  (half-dim)
        ChordLabel(B, dim, dom7)     → "Bdim7"  (fully dim)

    display_name is injective over VALID_COMBOS. Invalid combos (e.g. major + min7) render as
    "C[major+min7]" so they are visible instead of silently looking like another chord.
    """
    root:      RootNote
    quality:   ChordQuality
    extension: ChordExtension = ChordExtension.none

    @property
    def is_valid(self) -> bool:
        """True if (quality, extension) is a real chord (member of VALID_COMBOS)."""
        return (self.quality, self.extension) in VALID_COMBOS

    @property
    def display_name(self) -> str:
        root = self.root.value
        suffix = CHORD_SUFFIX.get((self.quality, self.extension))
        if suffix is None:
            return f"{root}[{self.quality.value}+{self.extension.value}]"
        return f"{root}{suffix}"

    def pitch_shift(self, semitones: int) -> ChordLabel:
        """Pitch shift root by n semitones, quality + extension unchanged."""
        new_idx = (RootNote.semitone_index(self.root) + semitones) % 12
        return ChordLabel(
            root=RootNote.from_semitone(new_idx),
            quality=self.quality,
            extension=self.extension,
        )

    def __str__(self) -> str:
        return self.display_name