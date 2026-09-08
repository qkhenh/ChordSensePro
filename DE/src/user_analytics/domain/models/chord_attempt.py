"""ChordAttempt entity: one practice attempt — user plays a chord, system compares to target."""
from __future__ import annotations
from dataclasses import dataclass

from src.shared.domain.base_model import BaseEntity


@dataclass
class ChordAttempt(BaseEntity):
    """A single chord recognition attempt during a Practice Mode session.

    Created each time a user plays a chord and the system evaluates it.
    Raw attempts are aggregated nightly by dag_analytics_rollup → ChordMastery.

    Fields:
        user_id:        User who made the attempt.
        session_id:     Groups all attempts in one practice session.
        target_chord:   The chord the UI asked the user to play (e.g. "Am7").
        detected_chord: What MERT predicted from the user's audio (e.g. "Am").
        root_conf:      Confidence score from Root head (0-1).
        quality_conf:   Confidence score from Quality head (0-1).
        extension_conf: Confidence score from Extension head (0-1).
        is_correct:     True when detected_chord == target_chord.
    """

    user_id:        str   = ""
    session_id:     str   = ""
    target_chord:   str   = ""
    detected_chord: str   = ""
    root_conf:      float = 0.0
    quality_conf:   float = 0.0
    extension_conf: float = 0.0
    is_correct:     bool  = False

    @property
    def avg_confidence(self) -> float:
        """Average confidence across all 3 heads."""
        return (self.root_conf + self.quality_conf + self.extension_conf) / 3
