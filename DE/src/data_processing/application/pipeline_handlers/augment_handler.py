"""AugmentHandler — label-aware audio augmentation for training data.

Position in training pipeline:
    AnnotationHandler → [AugmentHandler] → FeatureHandler

Applies:
  1. Pitch-shift ×12 keys (semitone steps -5 to +6, covers all 12 keys)
  2. Noise injection (Gaussian) for select variants
  3. Reverb simulation (simple IR convolution) for select variants

Oversampling strategy for class imbalance (per 18_DE_AI_chordsense.md):
  Basic chords (major/minor/no ext):    factor = 1   → ×12 keys only
  Semi-extended (maj7/min7/dom7/dim7):  factor = 2   → ×12 keys + noise variants
  Extended (b9, #9, 11th, 13th, alt):  factor = 4   → ×12 keys + noise + reverb variants

Total output per input segment:
  basic:    12 segments
  semi-ext: 24 segments
  extended: 48 segments

Skipped (pass-through) if segments have no chord_label — inference pipeline safe.
"""
from __future__ import annotations
import copy
import io

from src.data_processing.domain.models.base_handler import BaseProcessingHandler
from src.data_processing.domain.models.processed_audio import ProcessedAudio
from src.data_processing.domain.models.audio_segment import AudioSegment

# Harte extension tokens that qualify as "extended" (rare, need oversampling)
_EXTENDED_TOKENS = frozenset({
    "b5", "#5",
    "b9", "nat9", "#9",
    "nat11", "#11",
    "nat13", "b13",
    "alt", "add9",
})

# Harte extension tokens that qualify as "semi-extended"
_SEMI_EXTENDED_TOKENS = frozenset({
    "maj7", "min7", "7", "dim7", "hdim7",
})

# Oversampling factors — tune these to reduce class imbalance
# Current dataset ratio estimate: basic ~85%, semi-ext ~12%, extended ~3%
# Increase these if extended chords still underperform after training
OVERSAMPLE_BASIC        = 1   # major/minor → ×12 keys only
OVERSAMPLE_SEMI_EXT     = 2   # maj7/min7/dom7 → ×12 + noise variants
OVERSAMPLE_EXTENDED     = 4   # b9/#9/11th/13th/alt → ×12 + noise + reverb

# 12 semitone shifts covering all keys: C(0) → B(11)
_PITCH_SHIFTS = list(range(-5, 7))  # -5, -4, ..., 0, ..., +6

_ROOT_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_ROOT_INDEX  = {r: i for i, r in enumerate(_ROOT_NOTES)}
_ROOT_ALT    = {"Db": 1, "Eb": 3, "Gb": 6, "Ab": 8, "Bb": 10}  # enharmonic aliases


class AugmentHandler(BaseProcessingHandler):
    """Label-aware pitch-shift + noise/reverb augmentation for training pipeline."""

    def handle(self, data: ProcessedAudio) -> ProcessedAudio:
        if data.is_failed:
            return data

        # No chord labels → inference pipeline, skip silently
        if not any(s.chord_label for s in data.segments):
            return self._call_next(data)

        augmented: list[AudioSegment] = []
        for segment in data.segments:
            augmented.extend(self._augment_segment(segment))

        data.segments = augmented
        return self._call_next(data)

    def _augment_segment(self, segment: AudioSegment) -> list[AudioSegment]:
        """Generate augmented variants for one segment."""
        label  = segment.chord_label or "N"
        factor = self._oversample_factor(label)
        result: list[AudioSegment] = []

        # Step 1: ×12 pitch-shift variants
        for shift in _PITCH_SHIFTS:
            shifted_audio = self._pitch_shift(segment.audio_data, segment.sample_rate, shift)
            new_label = self._shift_label(label, shift)
            result.append(self._make_segment(segment, shifted_audio, new_label))

        # Step 2: extra noise/reverb copies for semi-extended and extended
        if factor >= 2:
            for shift in _PITCH_SHIFTS[:6]:   # first 6 keys get noise variant
                noisy = self._add_noise(
                    self._pitch_shift(segment.audio_data, segment.sample_rate, shift)
                )
                result.append(self._make_segment(segment, noisy, self._shift_label(label, shift)))

        if factor >= 4:
            for shift in _PITCH_SHIFTS[:6]:   # first 6 keys get reverb variant
                reverbed = self._add_reverb(
                    self._pitch_shift(segment.audio_data, segment.sample_rate, shift)
                )
                result.append(self._make_segment(segment, reverbed, self._shift_label(label, shift)))

        return result

    # ── Augmentation ops ──────────────────────────────────────────────────────

    def _pitch_shift(self, audio_bytes: bytes, sr: int, n_steps: int) -> bytes:
        """Pitch-shift audio by n_steps semitones. Returns WAV bytes."""
        if n_steps == 0:
            return audio_bytes
        import librosa
        import soundfile as sf
        import numpy as np
        audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=sr, mono=True)
        shifted   = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
        return self._to_bytes(shifted, sr)

    def _add_noise(self, audio_bytes: bytes, snr_db: float = 20.0) -> bytes:
        """Add Gaussian noise at specified SNR (dB)."""
        import numpy as np
        import librosa
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
        signal_power = np.mean(audio ** 2)
        noise_power  = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio)).astype(np.float32)
        return self._to_bytes(audio + noise, sr)

    def _add_reverb(self, audio_bytes: bytes, decay: float = 0.3) -> bytes:
        """Simple exponential decay reverb simulation."""
        import numpy as np
        import librosa
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
        ir_len = int(sr * 0.5)   # 500ms IR
        ir = np.exp(-decay * np.linspace(0, 1, ir_len)).astype(np.float32)
        reverbed = np.convolve(audio, ir, mode="full")[: len(audio)]
        reverbed /= (np.max(np.abs(reverbed)) + 1e-9)   # normalize
        return self._to_bytes(reverbed.astype(np.float32), sr)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _to_bytes(audio, sr: int) -> bytes:
        """Convert numpy float32 array → WAV bytes."""
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV")
        return buf.getvalue()

    @staticmethod
    def _make_segment(original: AudioSegment, audio_bytes: bytes, chord_label: str) -> AudioSegment:
        """Clone a segment with new audio_data and chord_label."""
        seg = copy.copy(original)
        seg.audio_data  = audio_bytes
        seg.chord_label = chord_label
        # Clear features — will be re-extracted by FeatureHandler
        seg.chroma_cqt = seg.chroma_cens = seg.hpss_harmonic = seg.mel_high = None
        return seg

    @staticmethod
    def _oversample_factor(harte_label: str) -> int:
        """Return oversampling factor based on chord complexity."""
        if harte_label in ("N", "X", ""):
            return OVERSAMPLE_BASIC
        parts = harte_label.split(":")
        ext   = parts[1] if len(parts) > 1 else ""
        if any(tok in ext for tok in _EXTENDED_TOKENS):
            return OVERSAMPLE_EXTENDED
        if any(tok in ext for tok in _SEMI_EXTENDED_TOKENS):
            return OVERSAMPLE_SEMI_EXT
        return OVERSAMPLE_BASIC

    @staticmethod
    def _shift_label(harte_label: str, n_steps: int) -> str:
        """Transpose root note in Harte label by n_steps semitones.

        'C:maj7' + 2 → 'D:maj7'
        'N' stays 'N' (no chord)
        """
        if harte_label in ("N", "X", "") or n_steps == 0:
            return harte_label

        parts = harte_label.split(":", 1)
        root  = parts[0]
        rest  = (":" + parts[1]) if len(parts) > 1 else ""

        # Resolve enharmonic alias → semitone index
        if root in _ROOT_ALT:
            idx = _ROOT_ALT[root]
        elif root in _ROOT_INDEX:
            idx = _ROOT_INDEX[root]
        else:
            return harte_label  # unrecognized root — return unchanged

        new_idx  = (idx + n_steps) % 12
        new_root = _ROOT_NOTES[new_idx]
        return f"{new_root}{rest}"
