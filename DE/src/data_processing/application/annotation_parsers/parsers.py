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
            ns = annotation.get("namespace", "")
            if ns not in ("chord", "chord_harte"):
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


class JaahParser:
    """Parse JAAH custom JSON format — parts[].beats[] + chords[].

    JAAH format (NOT standard JAMS):
        {
          "artist": "John Coltrane", "title": "Giant Steps",
          "metre": "4/4", "duration": 287.4,
          "parts": [
            {
              "name": "Head - AA",
              "beats": [0.25, 0.46, 0.67, ...],  // timestamps in seconds
              "chords": [
                "|B:maj7 D:7 |G:maj7 Bb:7 |Eb:maj7 |A:min7 D:7 |...",
                "|Eb:maj7 |A:min7 D:7 |G:maj7 |C#:min7 F#:7 |..."
              ]
            }, ...
          ]
        }

    Each chord line represents bars separated by '|'.
    Beats are distributed evenly across chords within each bar.
    """

    def parse(self, path: Path) -> ChordTimeline:
        """Parse JAAH JSON → chord timeline."""
        import json
        data = json.loads(path.read_text(encoding="utf-8"))
        timeline: ChordTimeline = []

        parts = data.get("parts", [])
        metre = data.get("metre", "4/4")
        beats_per_bar = int(metre.split("/")[0]) if "/" in str(metre) else 4

        for part in parts:
            beats  = part.get("beats", [])
            chords = part.get("chords", [])
            if not beats or not chords:
                continue

            # Parse all chord symbols from bar lines
            all_chords = self._parse_bars(chords)
            if not all_chords:
                continue

            # Distribute chords across beats:
            # Each bar has beats_per_bar beats. Each chord in a bar gets
            # equal share of beats within that bar.
            timeline.extend(self._align_chords_to_beats(all_chords, beats, beats_per_bar))

        # Deduplicate overlapping regions from multiple parts
        # (e.g. Head and Solo often repeat same chord progression)
        # Keep earliest occurrence for each time range
        if timeline:
            timeline = sorted(timeline, key=lambda x: x[0])
            deduped: ChordTimeline = [timeline[0]]
            for entry in timeline[1:]:
                prev = deduped[-1]
                if entry[0] >= prev[1]:  # no overlap
                    deduped.append(entry)
            timeline = deduped

        return timeline

    @staticmethod
    def _parse_bars(chord_lines: list[str]) -> list[list[str]]:
        """Parse chord bar strings → list of bars, each bar = list of chord symbols.

        '|B:maj7 D:7 |G:maj7 Bb:7 |Eb:maj7 |' → [['B:maj7', 'D:7'], ['G:maj7', 'Bb:7'], ['Eb:maj7']]
        """
        bars: list[list[str]] = []
        for line in chord_lines:
            # Split by '|', filter empty
            raw_bars = [b.strip() for b in line.split("|") if b.strip()]
            for bar_str in raw_bars:
                symbols = [s for s in bar_str.split() if s]
                if symbols:
                    bars.append(symbols)
        return bars

    @staticmethod
    def _align_chords_to_beats(
        bars: list[list[str]],
        beats: list[float],
        beats_per_bar: int,
    ) -> ChordTimeline:
        """Align parsed chord bars to beat timestamps.

        Each bar consumes beats_per_bar beats.
        Within a bar, beats are distributed evenly across chords.
        """
        timeline: ChordTimeline = []
        beat_idx = 0

        for bar_chords in bars:
            if beat_idx >= len(beats):
                break

            # How many beats this bar uses
            bar_beats = min(beats_per_bar, len(beats) - beat_idx)
            if bar_beats <= 0:
                break

            # Distribute beats across chords in this bar
            n_chords = len(bar_chords)
            beats_per_chord = max(1, bar_beats // n_chords)

            for chord_idx, chord_label in enumerate(bar_chords):
                start_beat = beat_idx + chord_idx * beats_per_chord
                end_beat   = beat_idx + (chord_idx + 1) * beats_per_chord

                if start_beat >= len(beats):
                    break

                start_time = beats[start_beat]
                end_time   = beats[min(end_beat, len(beats) - 1)]

                # Use next beat as end if available
                if end_beat < len(beats):
                    end_time = beats[end_beat]
                elif end_beat == len(beats) and end_beat > 0:
                    # Last chord: extend by average beat interval
                    avg_interval = (beats[-1] - beats[0]) / max(1, len(beats) - 1)
                    end_time = beats[-1] + avg_interval

                if end_time > start_time:
                    timeline.append((start_time, end_time, chord_label))

            beat_idx += bar_beats

        return timeline


def get_parser(annotation_path: Path) -> JamsParser | HarteParser | RwcParser | JaahParser:
    """Return the correct parser based on file extension and content.

    JAAH detection: .json files with 'parts' key → JaahParser
    Standard JAMS:  .jams files → JamsParser
    Harte .lab:     .lab/.txt → HarteParser
    RWC CSV:        .csv → RwcParser
    """
    ext = annotation_path.suffix.lower()

    if ext == ".json":
        # Detect JAAH format by checking for 'parts' key
        import json
        try:
            data = json.loads(annotation_path.read_text(encoding="utf-8"))
            if "parts" in data:
                return JaahParser()
        except Exception:
            pass
        # Fall back to JAMS parser (some JAMS files use .json extension)
        return JamsParser()

    if ext == ".jams":
        return JamsParser()
    elif ext in (".lab", ".txt"):
        return HarteParser()
    elif ext == ".csv":
        return RwcParser()
    raise ValueError(f"Unsupported annotation format: {ext}. Expected .jams, .json, .lab, .txt, .csv")
