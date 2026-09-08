"""Annotation parsers — parse chord labels from dataset annotation files.

Each parser returns a list of (start_sec, end_sec, chord_label) tuples
in Harte notation. Used by AnnotationHandler to assign chord_label to segments.

Supported formats:
  JamsParser  — McGill Billboard (JAMS JSON), JAAH (JAMS JSON)
  HarteParser — ChoCo Corpus (.lab / .txt in Harte notation)
  RwcParser   — RWC Popular Music (beat-level CSV)
"""
from __future__ import annotations
from pathlib import Path

# Type alias for chord timeline: list of (start_sec, end_sec, harte_label)
ChordTimeline = list[tuple[float, float, str]]


class JamsParser:
    """Parse JAMS format chord annotations (McGill Billboard, JAAH).

    JAMS spec: https://jams.readthedocs.io/
    Chord namespace: annotation['namespace'] == 'chord'
    Each observation: {'time': float, 'duration': float, 'value': str}
    Value is Harte notation: 'C:major', 'A:min7', 'N' (no chord)
    """

    def parse(self, path: Path) -> ChordTimeline:
        """Parse JAMS file → chord timeline."""
        import json
        data = json.loads(path.read_text(encoding="utf-8"))
        timeline: ChordTimeline = []

        for annotation in data.get("annotations", []):
            if annotation.get("namespace") != "chord":
                continue
            for obs in annotation.get("data", []):
                start = float(obs.get("time", 0.0))
                dur   = float(obs.get("duration", 0.0))
                label = str(obs.get("value", "N"))
                if dur > 0:
                    timeline.append((start, start + dur, label))
            break  # use first chord annotation track

        return sorted(timeline, key=lambda x: x[0])


class HarteParser:
    """Parse Harte notation .lab files (ChoCo Corpus).

    Format (space or tab separated):
        start_sec  end_sec  chord_label
        0.000      2.000    C:maj
        2.000      4.000    A:min7
        4.000      5.500    N

    Harte notation: Root:quality(extensions)
    'N' = no chord / silence
    """

    def parse(self, path: Path) -> ChordTimeline:
        """Parse Harte .lab file → chord timeline."""
        timeline: ChordTimeline = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                start = float(parts[0])
                end   = float(parts[1])
                label = parts[2]
                timeline.append((start, end, label))
            except ValueError:
                continue
        return sorted(timeline, key=lambda x: x[0])


class RwcParser:
    """Parse RWC Popular Music beat-level CSV chord annotations.

    Format (tab separated):
        beat_time  chord_label
        0.000      C
        0.500      Am
        1.000      F

    RWC uses simplified notation (no Harte quality suffix for basic chords).
    This parser normalizes to Harte: 'C' → 'C:maj', 'Am' → 'A:min'
    """

    # Simple suffix → Harte quality mapping
    _SUFFIX_MAP = {
        "m":   "min", "min": "min", "M":    "maj", "maj": "maj",
        "7":   "7",   "m7":  "min7", "M7":  "maj7", "dim": "dim",
        "aug": "aug", "sus": "sus4", "sus4": "sus4", "sus2": "sus2",
        "":    "maj",
    }

    def parse(self, path: Path) -> ChordTimeline:
        """Parse RWC CSV → chord timeline in Harte notation."""
        rows: list[tuple[float, str]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split()
            if len(parts) < 2:
                continue
            try:
                rows.append((float(parts[0]), parts[1]))
            except ValueError:
                continue

        timeline: ChordTimeline = []
        for i, (start, label) in enumerate(rows):
            end = rows[i + 1][0] if i + 1 < len(rows) else start + 2.0
            harte = self._to_harte(label)
            timeline.append((start, end, harte))
        return timeline

    def _to_harte(self, label: str) -> str:
        """Normalize RWC chord label → Harte notation."""
        if label in ("N", "X", ""):
            return "N"
        # Parse root and suffix: Am7 → A + m7
        root = ""
        for i in range(min(3, len(label))):
            if label[i] in ("A", "B", "C", "D", "E", "F", "G"):
                root = label[i]
                if i + 1 < len(label) and label[i + 1] in ("#", "b"):
                    root += label[i + 1]
                    suffix = label[i + 2:]
                else:
                    suffix = label[i + 1:]
                break
        else:
            return "N"
        quality = self._SUFFIX_MAP.get(suffix, "maj")
        return f"{root}:{quality}"


def get_parser(annotation_path: Path) -> JamsParser | HarteParser | RwcParser:
    """Return the correct parser based on file extension."""
    ext = annotation_path.suffix.lower()
    if ext == ".jams":
        return JamsParser()
    elif ext in (".lab", ".txt"):
        return HarteParser()
    elif ext == ".csv":
        return RwcParser()
    raise ValueError(f"Unsupported annotation format: {ext}. Expected .jams, .lab, .txt, .csv")
