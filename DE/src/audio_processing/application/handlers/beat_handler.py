from __future__ import annotations
from pathlib import Path

import librosa
import numpy as np

from src.shared.domain.processing_result import ProcessingResult


class BeatHandler:
    """Handler: WAV → beat timestamps via librosa beat tracker.

    Replaces madmom (unmaintained since 2022, build issues on Python 3.12+).
    librosa.beat.beat_track uses a dynamic programming beat tracker — sufficient
    for thesis purposes and handles most pop/rock/jazz without issues.

    Output: list[float] — beat_times in seconds.
    Example: [0.0, 0.5, 1.0, 1.5, ...] for 120 BPM song
    """

    TARGET_SR = 44100

    def run(self, wav_path: Path) -> ProcessingResult[list[float]]:
        """Detect beats and return timestamps in seconds.

        Args:
            wav_path: Path to WAV 44100Hz (after Demucs separation)

        Returns:
            ProcessingResult[list[float]] — beat timestamps in seconds
        """
        try:
            audio, sr = librosa.load(str(wav_path), sr=self.TARGET_SR, mono=True)
            tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

            if not beat_times:
                return ProcessingResult.err("No beats detected — audio may be silent or arrhythmic")

            return ProcessingResult.ok(beat_times)

        except Exception as e:
            return ProcessingResult.err(f"Beat tracking failed: {e}")

    def estimate_bpm(self, beat_times: list[float]) -> float:
        """Calculate average BPM from inter-beat intervals.

        Args:
            beat_times: Beat timestamps in seconds

        Returns:
            Average BPM, or 0.0 if not enough beats
        """
        if len(beat_times) < 2:
            return 0.0
        intervals = np.diff(beat_times)
        return float(60.0 / np.mean(intervals))