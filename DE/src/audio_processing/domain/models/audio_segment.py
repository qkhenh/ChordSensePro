from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from src.shared.domain.base_model import BaseEntity
from src.shared.domain.value_objects import ChromaVector

@dataclass
class AudioSegment(BaseEntity):
    """Each audio segment lasts 2s - thứ nguyên của pipeline
    
    Create from SegmentHandler after Demucs + Beat Tracking.
    Pass to FeatureHandler to extract features
    After that, MERT receives input from here. 
    """
    
    song_id : str = ""
    segment_idx : int = 0
    start_ms : float = 0.0 
    end_ms : float = 0.0
    sample_rate : int = 44100  #! DownloadHandler need to resample down to 44100Hz
    audio_data : bytes = field(default_factory=bytes)
    
    chroma_cqt : list[float] | None = None
    chroma_cens : list[float] | None = None
    mel_high : list[float] | None = None
    
    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms
    
    @property
    def has_features(self) -> bool:
        return self.chroma_cqt is not None
    
    