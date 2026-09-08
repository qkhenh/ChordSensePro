"""SongAnalysis entity: tracks the full chord recognition result for one user-submitted song."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum

from src.shared.domain.base_model import AuditedEntity

class AnalysisStatus(str, Enum):
    PENDING =  "pending"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"
    
@dataclass
class SongAnalysis(AuditedEntity):
    """Represents the full analysis result of one song.
    
    Created when user submits a Youtube URL or audio file
    Populated step by step as pipeline progresses 
    Final result stored in PostgreSQL song_analysis table 

    Args:
        AuditedEntity (_type_): _description_
    """
    
    user_id : str = ""
    source_url : str = ""
    song_title : str = ""
    status : AnalysisStatus = AnalysisStatus.PENDING
    
    detected_key : str | None = None 
    tempo_bpm : float | None = None
    
    chord_timeline : list[dict] = field(default_factory=list)
    chord_sheet : dict = field(default_factory=dict)
    learning_plan : list[dict] = field(default_factory=list)
    error_message : str | None = None 
    
    def mark_processing(self) -> None: 
        self.status = AnalysisStatus.PROCESSING
        self.touch()
        
    def mark_done(
        self,
        key: str,
        bpm: float, 
        timeline: list[dict],
        sheet: dict,
        plan: list[dict],
    ) -> None:
        self.detected_key = key
        self.tempo_bpm = bpm
        self.chord_timeline = timeline
        self.chord_sheet = sheet
        self.learning_plan = plan
        self.status = AnalysisStatus.DONE
        self.touch()
        
    def mark_error(self, message: str) -> None:
        self.error_message = message
        self.status = AnalysisStatus.ERROR
        self.touch()
        
    @property
    def is_done(self) -> bool:
        return self.status == AnalysisStatus.DONE  