"""
shared/domain/value_objects.py

Value Objects cốt lõi của ChordSense Pro.
Immutable, so sánh bằng value, không phụ thuộc infrastructure.

Naming convention: theo Chordie.com (chord notation phổ biến nhất với musician).
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class RootNote(str, Enum):
    """12 chromatic pitch classes — dùng sharp làm canonical form.

    Enharmonic disambiguation (Db vs C#) xử lý ở post-processing
    layer sau khi biết Key của bài, không phải ở đây.
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
        """Dùng cho pitch shift: (index + n_steps) % 12 → RootNote"""
        return list(cls)[idx % 12]


class ChordQuality(str, Enum):
    """5 chord qualities — 'aug' bị loại vì quy về major + sharp5."""
    major     = "major"
    minor     = "minor"
    dominant  = "dominant"   # implies b7, dùng cho G7, G7#9, G7alt...
    dim       = "diminished" # 1-b3-b5
    sus       = "suspended"  # sus2 (1-2-5) hoặc sus4 (1-4-5)


class ChordExtension(str, Enum):
    """15 extension classes — theo Chordie.com naming convention."""
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
    alt    = "alt"    # G7alt = b9+#9+#11+b13 cùng lúc


@dataclass(frozen=True)
class ChromaVector:
    """12-dim pitch class energy vector. Index 0=C, 1=C#, ..., 11=B."""
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.values) != 12:
            raise ValueError(f"ChromaVector cần 12 dims, got {len(self.values)}")
        if not all(0.0 <= v <= 1.0 for v in self.values):
            raise ValueError("ChromaVector values phải trong [0.0, 1.0]")

    @classmethod
    def from_list(cls, values: Sequence[float]) -> ChromaVector:
        return cls(values=tuple(values))

    def dominant_pitch(self) -> RootNote:
        """Pitch class có energy cao nhất — heuristic root detection."""
        idx = max(range(12), key=lambda i: self.values[i])
        return RootNote.from_semitone(idx)


@dataclass(frozen=True)
class ConfidenceScore:
    """Per-head confidence từ 3 classification heads của model."""
    root: float       # Head 1 [0, 1]
    quality: float    # Head 2 [0, 1]
    extension: float  # Head 3 [0, 1]

    def __post_init__(self) -> None:
        for name, val in [("root", self.root), ("quality", self.quality), ("extension", self.extension)]:
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"ConfidenceScore.{name} phải trong [0.0, 1.0]")

    @property
    def overall(self) -> float:
        """Weighted overall — root quan trọng nhất, extension khó nhất nên weight thấp.
        Tuneable sau khi có model thật và validation set."""
        return 0.4 * self.root + 0.35 * self.quality + 0.25 * self.extension

    @property
    def is_reliable(self) -> bool:
        """True nếu đủ tự tin để hiển thị cho user.
        Threshold empirical — sẽ tune sau khi train model."""
        return self.root >= 0.6 and self.quality >= 0.55


@dataclass(frozen=True)
class ChordLabel:
    """Full chord label — output cuối của model.

    Naming theo Chordie.com convention.
    Ví dụ:
        ChordLabel(A, minor, min7)   → "Am7"
        ChordLabel(C, major, sharp5) → "Caug"
        ChordLabel(G, dominant, sharp9) → "G7#9"
        ChordLabel(B, dim, min7)     → "Bm7b5"  (half-dim)
        ChordLabel(B, dim, dom7)     → "Bdim7"  (fully dim)
    """
    root:      RootNote
    quality:   ChordQuality
    extension: ChordExtension = ChordExtension.none

    @property
    def display_name(self) -> str:
        root = self.root.value

        # --- Xử lý đặc biệt: dim + extension (Chordie convention) ---
        if self.quality == ChordQuality.dim:
            if self.extension == ChordExtension.min7:
                return f"{root}m7b5"   # half-diminished
            if self.extension == ChordExtension.dom7:
                return f"{root}dim7"   # fully diminished
            if self.extension == ChordExtension.none:
                return f"{root}dim"

        # --- Quality suffix ---
        q = {
            ChordQuality.major:    "",
            ChordQuality.minor:    "m",
            ChordQuality.dominant: "",
            ChordQuality.dim:      "dim",
            ChordQuality.sus:      "sus",
        }[self.quality]

        # --- Extension suffix ---
        e = {
            ChordExtension.none:   "",
            ChordExtension.flat5:  "b5",
            ChordExtension.sharp5: "aug",       # Caug quen hơn C#5 trên Chordie
            ChordExtension.maj7:   "maj7",
            ChordExtension.min7:   "7",         # "m" đã có trong q rồi → Am7
            ChordExtension.dom7:   "7",
            ChordExtension.add9:   "add9",
            ChordExtension.flat9:  "7b9",
            ChordExtension.nat9:   "9",
            ChordExtension.sharp9: "7#9",
            ChordExtension.nat11:  "11",
            # sharp11: major → "maj7#11", dominant → "7#11" (Chordie style)
            ChordExtension.sharp11: "maj7#11" if self.quality == ChordQuality.major else "7#11",
            ChordExtension.nat13:  "13",
            ChordExtension.flat13: "7b13",
            ChordExtension.alt:    "7alt",
        }[self.extension]

        return f"{root}{q}{e}"

    def pitch_shift(self, semitones: int) -> ChordLabel:
        """Augmentation: shift root n semitones, quality + extension không đổi."""
        new_idx = (RootNote.semitone_index(self.root) + semitones) % 12
        return ChordLabel(
            root=RootNote.from_semitone(new_idx),
            quality=self.quality,
            extension=self.extension,
        )

    def __str__(self) -> str:
        return self.display_name