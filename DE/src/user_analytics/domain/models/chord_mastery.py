"""ChordMastery entity: daily aggregated accuracy per (user, chord) — used to build learning_plan."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date


MASTERY_THRESHOLD = 0.80   # 80% rolling_accuracy_3d → is_mastered = True


@dataclass
class ChordMastery:
    """Daily mastery record for one chord per user.

    Not a BaseEntity — uses composite natural key (user_id, chord, date).
    Computed nightly by dag_analytics_rollup from chord_attempts.

    Fields:
        user_id:             User identifier.
        chord:               Chord label, e.g. "Am7", "Cmaj7#11".
        date:                The date this record covers (UTC).
        accuracy_today:      Fraction correct today (attempts today).
        rolling_accuracy_3d: 3-day rolling accuracy (primary mastery signal).
        is_mastered:         True when rolling_accuracy_3d >= 80%.
    """

    user_id:             str
    chord:               str
    date:                date
    accuracy_today:      float = 0.0
    rolling_accuracy_3d: float = 0.0
    is_mastered:         bool  = False

    def recompute_mastery(self) -> None:
        """Re-evaluate is_mastered based on rolling accuracy and threshold."""
        self.is_mastered = self.rolling_accuracy_3d >= MASTERY_THRESHOLD

    @property
    def needs_practice(self) -> bool:
        """True when the chord is not yet mastered."""
        return not self.is_mastered
