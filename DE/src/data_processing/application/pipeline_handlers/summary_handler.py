"""SummaryHandler — aggregates segment-level chord labels into song-level summary.

Position in training pipeline:
    AnnotationHandler → [SummaryHandler] → AugmentHandler → FeatureHandler

Reads each segment's chord_label (set by AnnotationHandler) and produces:
  1. chord_timeline — ordered list of chord events with timestamps
  2. detected_key — estimated musical key from chord root distribution
  3. chord_sheet — simplified chord chart (deduplicated, grouped)

Skipped (pass-through) if no segments have chord_label — inference pipeline safe.
"""
from __future__ import annotations
from collections import Counter

from src.data_processing.domain.models.base_handler import BaseProcessingHandler
from src.data_processing.domain.models.processed_audio import ProcessedAudio


# Note → semitone index (for key detection)
_ROOT_INDEX = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

# Major key → characteristic chord set (simplified music theory)
# If root appears most, and its IV and V also appear → likely that key
_MAJOR_KEYS = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


class SummaryHandler(BaseProcessingHandler):
    """Aggregate segment chord_labels into song-level chord_timeline, detected_key, chord_sheet."""

    def handle(self, data: ProcessedAudio) -> ProcessedAudio:
        if data.is_failed:
            return data

        # No chord labels → inference pipeline, skip silently
        labeled_segments = [s for s in data.segments if s.chord_label and s.chord_label != "N"]
        if not labeled_segments:
            return self._call_next(data)

        # 1. Build chord_timeline from segment labels
        data.chord_timeline = self._build_timeline(data.segments)

        # 2. Detect key from chord root distribution
        data.detected_key = self._detect_key(data.chord_timeline)

        # 3. Build simplified chord sheet
        data.chord_sheet = self._build_chord_sheet(data.chord_timeline)

        return self._call_next(data)

    @staticmethod
    def _build_timeline(segments: list) -> list[dict]:
        """Collect segment chord_labels into a deduplicated chord timeline.

        Consecutive segments with the same chord are merged into one event.
        Output: [{time_ms: float, duration_ms: float, chord: str}, ...]
        """
        timeline: list[dict] = []

        for seg in sorted(segments, key=lambda s: s.start_ms):
            label = seg.chord_label or "N"
            if label == "N":
                continue

            # Merge with previous if same chord
            if timeline and timeline[-1]["chord"] == label:
                timeline[-1]["duration_ms"] = seg.end_ms - timeline[-1]["time_ms"]
            else:
                timeline.append({
                    "time_ms":     seg.start_ms,
                    "duration_ms": seg.end_ms - seg.start_ms,
                    "chord":       label,
                })

        return timeline

    @staticmethod
    def _detect_key(timeline: list[dict]) -> str:
        """Estimate musical key from chord root distribution.

        Simple heuristic: most frequent root note = likely key.
        If the quality is mostly minor → append 'm' (e.g. "Am").
        """
        if not timeline:
            return ""

        root_counts: Counter = Counter()
        quality_counts: Counter = Counter()

        for event in timeline:
            chord = event["chord"]
            parts = chord.split(":", 1)
            root  = parts[0]
            qual  = parts[1] if len(parts) > 1 else "maj"

            if root in _ROOT_INDEX:
                root_counts[root] += 1
                quality_counts[qual] += 1

        if not root_counts:
            return ""

        # Most common root
        top_root, _ = root_counts.most_common(1)[0]

        # Check if mostly minor quality → minor key
        minor_count = sum(v for k, v in quality_counts.items() if "min" in k)
        major_count = sum(v for k, v in quality_counts.items() if "maj" in k or k in ("", "7"))
        total = minor_count + major_count

        if total > 0 and minor_count / total > 0.6:
            return f"{top_root}m"
        return top_root

    @staticmethod
    def _build_chord_sheet(timeline: list[dict]) -> dict:
        """Build a simplified chord sheet from chord_timeline.

        Groups chords into bars (4 events per bar approximation).
        Output: {"chords": ["C:maj", "Am:min", "F:maj", "G:7", ...]}
        """
        if not timeline:
            return {}

        # Deduplicate consecutive chords
        unique_chords: list[str] = []
        for event in timeline:
            chord = event["chord"]
            if not unique_chords or unique_chords[-1] != chord:
                unique_chords.append(chord)

        return {"chords": unique_chords}
