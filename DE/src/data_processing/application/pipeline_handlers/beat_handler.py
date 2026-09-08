"""Handler: detects beat timestamps and estimates BPM using librosa beat tracker."""
from __future__ import annotations

from src.data_processing.domain.models.base_handler import BaseProcessingHandler
from src.data_processing.domain.models.processed_audio import ProcessedAudio


class BeatHandler(BaseProcessingHandler):
    """Handler: WAV → beat timestamps via librosa beat tracker.

    Replaces madmom (unmaintained since 2022, build issues on Python 3.12+).
    librosa.beat.beat_track uses a dynamic programming beat tracker — sufficient
    for thesis purposes and handles most pop/rock/jazz without issues.

    Output: list[float] — beat_times in seconds.
    Example: [0.0, 0.5, 1.0, 1.5, ...] for 120 BPM song
    """

    TARGET_SR = 44100

    def handle(self, data: ProcessedAudio) -> ProcessedAudio:
        if data.is_failed:
            return data

        try:
            import librosa  # lazy — requires audio extras (Docker only)
            audio, sr = librosa.load(str(data.stem_path), sr=self.TARGET_SR, mono=True)
            tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

            if not beat_times:
                data.error = "No beats detected — audio may be silent or arrhythmic"
                return data

            data.beat_times = beat_times
            data.tempo_bpm  = self.estimate_bpm(beat_times)
            return self._call_next(data)

        except Exception as e:
            data.error = f"Beat tracking failed: {e}"
            return data

    def estimate_bpm(self, beat_times: list[float]) -> float:
        """Calculate average BPM from inter-beat intervals."""
        if len(beat_times) < 2:
            return 0.0
        import numpy as np  # lazy
        intervals = np.diff(beat_times)
        return float(60.0 / np.mean(intervals))
