"""Handler: extracts all 4 feature channels (chroma_cqt, chroma_cens, hpss_harmonic, mel_high) from AudioSegment bytes."""
from __future__ import annotations
import io

from src.data_processing.domain.models.base_handler import BaseProcessingHandler
from src.data_processing.domain.models.processed_audio import ProcessedAudio
from src.data_processing.domain.models.audio_segment import AudioSegment
from src.shared.domain.processing_result import ProcessingResult


class FeatureHandler(BaseProcessingHandler):
    """Handler: AudioSegment (raw bytes) → AudioSegment (with all 4 features filled).

    Extracts 4-channel multi-scale feature input per spec section 3.1 [5]:
      - chroma_cqt    (12-dim): pitch class energy via Constant-Q Transform
      - chroma_cens   (12-dim): noise-robust chroma, stable across timbres
      - hpss_harmonic (64-dim): mel spectrogram of HPSS harmonic component (chord body)
      - mel_high      (64-dim): mel at 2-8kHz, captures 9th/11th/13th extensions

    Total feature vector: 152-dim per segment, fed into MERT classification heads.
    """

    TARGET_SR   = 22050    # librosa default — downsampled from 44100 for feature extraction
    N_MELS      = 64
    HPSS_N_MELS = 64       # mel bins for HPSS harmonic component
    F_MIN_HIGH  = 2000.0   # Hz — lower bound of high-freq mel band
    F_MAX_HIGH  = 8000.0   # Hz — upper bound of high-freq mel band

    def handle(self, data: ProcessedAudio) -> ProcessedAudio:
        if data.is_failed:
            return data

        results = self.run_batch(data.segments)
        failed = [r for r in results if r.is_err]
        if failed:
            data.error = failed[0].error
            return data

        data.segments = [r.unwrap() for r in results]
        return self._call_next(data)

    def run(self, segment: AudioSegment) -> ProcessingResult[AudioSegment]:
        """Extract all 4 feature channels from segment audio bytes.

        Args:
            segment: AudioSegment with audio_data filled (from SegmentHandler)

        Returns:
            ProcessingResult[AudioSegment] — new instance with all features populated
        """
        if not segment.audio_data:
            return ProcessingResult.err(f"Segment {segment.segment_idx}: audio_data is empty")

        try:
            audio = self._load_audio(segment.audio_data)
            features = self._extract(audio)

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
                hpss_harmonic=features["hpss_harmonic"],
                mel_high=features["mel_high"],
            )
            return ProcessingResult.ok(updated)

        except Exception as e:
            return ProcessingResult.err(
                f"Feature extraction failed on segment {segment.segment_idx}: {e}"
            )

    def _load_audio(self, audio_bytes: bytes):
        """Load WAV bytes → numpy array at TARGET_SR (22050Hz)."""
        import librosa  # lazy
        buf = io.BytesIO(audio_bytes)
        audio, _ = librosa.load(buf, sr=self.TARGET_SR, mono=True)
        return audio

    def _extract(self, audio) -> dict[str, list[float]]:
        """Extract all 4 feature channels and return as Python lists."""
        import librosa  # lazy
        sr = self.TARGET_SR

        # Channel 1 — 12-dim: pitch class energy (primary root note signal)
        chroma_cqt = librosa.feature.chroma_cqt(y=audio, sr=sr)
        chroma_cqt_mean: list[float] = chroma_cqt.mean(axis=1).tolist()

        # Channel 2 — 12-dim: noise-robust chroma (stable across piano timbres)
        chroma_cens = librosa.feature.chroma_cens(y=audio, sr=sr)
        chroma_cens_mean: list[float] = chroma_cens.mean(axis=1).tolist()

        # Channel 3 — 64-dim: HPSS harmonic mel (pure chord body, percussive removed)
        harmonic, _ = librosa.effects.hpss(audio)
        mel_harmonic = librosa.feature.melspectrogram(
            y=harmonic, sr=sr, n_mels=self.HPSS_N_MELS,
        )
        hpss_harmonic_mean: list[float] = librosa.power_to_db(mel_harmonic).mean(axis=1).tolist()

        # Channel 4 — 64-dim: high-frequency mel (9th/11th/13th extension info)
        mel_high = librosa.feature.melspectrogram(
            y=audio, sr=sr,
            n_mels=self.N_MELS,
            fmin=self.F_MIN_HIGH,
            fmax=self.F_MAX_HIGH,
        )
        mel_high_mean: list[float] = librosa.power_to_db(mel_high).mean(axis=1).tolist()

        return {
            "chroma_cqt":    chroma_cqt_mean,
            "chroma_cens":   chroma_cens_mean,
            "hpss_harmonic": hpss_harmonic_mean,
            "mel_high":      mel_high_mean,
        }

    def run_batch(self, segments: list[AudioSegment]) -> list[ProcessingResult[AudioSegment]]:
        """Extract features for a list of segments. Returns one result per segment."""
        return [self.run(seg) for seg in segments]
