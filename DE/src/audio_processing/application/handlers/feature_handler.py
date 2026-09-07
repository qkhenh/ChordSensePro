from __future__ import annotations
import io

import librosa
import numpy as np

from src.shared.domain.processing_result import ProcessingResult
from src.audio_processing.domain.models.audio_segment import AudioSegment


class FeatureHandler:
    """Handler: AudioSegment (raw bytes) → AudioSegment (with features filled).

    Extracts 3 feature vectors per segment (per spec section 3.3):
      - chroma_cqt  (12-dim): pitch class energy via Constant-Q Transform
      - chroma_cens (12-dim): noise-robust chroma, stable across timbres
      - mel_high    (64-dim): mel spectrogram at 2-8kHz, captures 9th/11th/13th

    Returns the same AudioSegment with chroma_cqt, chroma_cens, mel_high filled.
    """

    TARGET_SR = 22050    # librosa default — resampled from 44100 for feature extraction
    N_MELS    = 64
    F_MIN     = 2000.0   # Hz — start of high-frequency mel band
    F_MAX     = 8000.0   # Hz — end of high-frequency mel band

    def run(self, segment: AudioSegment) -> ProcessingResult[AudioSegment]:
        """Extract features from segment audio bytes.

        Args:
            segment: AudioSegment with audio_data filled (from SegmentHandler)

        Returns:
            ProcessingResult[AudioSegment] — same segment with features populated
        """
        if not segment.audio_data:
            return ProcessingResult.err(f"Segment {segment.segment_idx}: audio_data is empty")

        try:
            audio = self._load_audio(segment.audio_data)
            features = self._extract(audio)

            # Dataclass is frozen=False here (BaseEntity uses @dataclass not frozen)
            # so we create a new instance with features filled
            updated = AudioSegment(
                id=segment.id,
                created_at=segment.created_at,
                song_id=segment.song_id,
                segment_idx=segment.segment_idx,
                start_ms=segment.start_ms,
                end_ms=segment.end_ms,
                sample_rate=segment.sample_rate,
                audio_data=segment.audio_data,
                chroma_cqt=features["chroma_cqt"],
                chroma_cens=features["chroma_cens"],
                mel_high=features["mel_high"],
            )
            return ProcessingResult.ok(updated)

        except Exception as e:
            return ProcessingResult.err(f"Feature extraction failed on segment {segment.segment_idx}: {e}")

    def _load_audio(self, audio_bytes: bytes) -> np.ndarray:
        """Load WAV bytes → numpy array at TARGET_SR (22050Hz)."""
        buf = io.BytesIO(audio_bytes)
        audio, _ = librosa.load(buf, sr=self.TARGET_SR, mono=True)
        return audio

    def _extract(self, audio: np.ndarray) -> dict[str, list[float]]:
        """Extract all 3 feature vectors and return as Python lists."""
        sr = self.TARGET_SR

        # 12-dim: pitch class energy (root note detection)
        chroma_cqt = librosa.feature.chroma_cqt(y=audio, sr=sr)
        chroma_cqt_mean = chroma_cqt.mean(axis=1).tolist()

        # 12-dim: noise-robust chroma (stable across different piano timbres)
        chroma_cens = librosa.feature.chroma_cens(y=audio, sr=sr)
        chroma_cens_mean = chroma_cens.mean(axis=1).tolist()

        # 64-dim: high-frequency mel (captures 9th/11th/13th harmonic content)
        mel_high = librosa.feature.melspectrogram(
            y=audio, sr=sr,
            n_mels=self.N_MELS,
            fmin=self.F_MIN,
            fmax=self.F_MAX,
        )
        mel_high_mean = librosa.power_to_db(mel_high).mean(axis=1).tolist()

        return {
            "chroma_cqt":  chroma_cqt_mean,
            "chroma_cens": chroma_cens_mean,
            "mel_high":    mel_high_mean,
        }

    def run_batch(self, segments: list[AudioSegment]) -> list[ProcessingResult[AudioSegment]]:
        """Extract features for a list of segments. Returns result per segment."""
        return [self.run(seg) for seg in segments]
