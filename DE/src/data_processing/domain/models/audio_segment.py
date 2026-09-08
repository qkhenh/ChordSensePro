"""AudioSegment entity: a 2-second audio chunk produced by SegmentHandler, passed to FeatureHandler for extraction."""
from __future__ import annotations
from dataclasses import dataclass, field

from src.shared.domain.base_model import BaseEntity

@dataclass
class AudioSegment(BaseEntity):
    """2-second audio chunk produced by SegmentHandler.

    Passed to FeatureHandler which fills all 4 feature channels:
      chroma_cqt    (12-dim) — pitch class energy via CQT
      chroma_cens   (12-dim) — noise-robust chroma
      hpss_harmonic (64-dim) — mel spectrogram of HPSS harmonic component
      mel_high      (64-dim) — mel at 2-8kHz for extension detection

    All features are None until FeatureHandler.run() is called.
    After that, MERT receives the 4-channel input from here.
    """
    
    song_id : str = ""
    segment_idx : int = 0
    start_ms : float = 0.0 
    end_ms : float = 0.0
    sample_rate : int = 44100  # YoutubeDownloader resamples to 44100Hz mono via ffmpeg
    audio_data : bytes = field(default_factory=bytes)
    
    chroma_cqt    : list[float] | None = None   # 12-dim
    chroma_cens   : list[float] | None = None   # 12-dim
    hpss_harmonic : list[float] | None = None   # 64-dim
    mel_high      : list[float] | None = None   # 64-dim
    chord_label   : str | None = None           # Harte notation, e.g. "C:maj7" — set by AnnotationHandler
    
    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms
    
    @property
    def has_features(self) -> bool:
        """True only when all 4 feature channels are populated."""
        return (
            self.chroma_cqt is not None
            and self.chroma_cens is not None
            and self.hpss_harmonic is not None
            and self.mel_high is not None
        )
    
    