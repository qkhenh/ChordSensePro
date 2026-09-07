from __future__ import annotations
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from src.shared.domain.processing_result import ProcessingResult
from src.audio_processing.domain.models.audio_segment import AudioSegment


class SegmentHandler:
    """Handler: WAV + beat_times → list of 2s AudioSegments aligned to beats.

    Segments are beat-aligned to avoid cutting mid-chord.
    Each segment is ~2s, with 0.5s overlap for real-time pipeline.
    """

    SEGMENT_DURATION = 2.0   # seconds
    OVERLAP = 0.5            # seconds overlap between segments

    def run(
        self,
        wav_path: Path,
        beat_times: list[float],
        song_id: str,
    ) -> ProcessingResult[list[AudioSegment]]:
        """Slice audio into beat-aligned 2s segments.

        Args:
            wav_path:   Path to WAV 44100Hz (after Demucs)
            beat_times: Beat timestamps in seconds (from BeatHandler)
            song_id:    Song identifier for tracing

        Returns:
            ProcessingResult[list[AudioSegment]]
        """
        try:
            audio, sr = librosa.load(str(wav_path), sr=44100, mono=True)
            total_duration = len(audio) / sr
            segments: list[AudioSegment] = []

            step = self.SEGMENT_DURATION - self.OVERLAP
            start = 0.0
            idx = 0

            while start + self.SEGMENT_DURATION <= total_duration:
                end = start + self.SEGMENT_DURATION
                start_sample = int(start * sr)
                end_sample   = int(end * sr)

                chunk = audio[start_sample:end_sample]
                audio_bytes = self._to_wav_bytes(chunk, sr)

                segments.append(AudioSegment(
                    song_id=song_id,
                    segment_idx=idx,
                    start_ms=start * 1000,
                    end_ms=end * 1000,
                    sample_rate=sr,
                    audio_data=audio_bytes,
                ))

                # Snap to nearest beat for next segment start
                start = self._snap_to_beat(start + step, beat_times)
                idx += 1

            if not segments:
                return ProcessingResult.err("No segments produced — audio too short")

            return ProcessingResult.ok(segments)

        except Exception as e:
            return ProcessingResult.err(f"Segmentation failed: {e}")

    def _snap_to_beat(self, t: float, beat_times: list[float]) -> float:
        """Snap timestamp to nearest beat within 0.1s tolerance."""
        beats = np.array(beat_times)
        diffs = np.abs(beats - t)
        nearest_idx = int(np.argmin(diffs))
        if diffs[nearest_idx] < 0.1:
            return float(beats[nearest_idx])
        return t

    def _to_wav_bytes(self, audio: np.ndarray, sr: int) -> bytes:
        """Convert numpy array to WAV bytes in memory (no temp file)."""
        import io
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV")
        return buf.getvalue()
