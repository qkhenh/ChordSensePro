"""AudioFile — request model representing one audio download job."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from src.shared.domain.base_model import BaseEntity


class AudioSource(str, Enum):
    """Supported audio source types."""
    YOUTUBE = "youtube"
    LOCAL   = "local"


@dataclass
class AudioFile(BaseEntity):
    """Represents one audio download request.

    Created by data_ingest, passed to AudioDispatcher.
    source_type drives which Downloader implementation is used.
    """
    source_url:  str         = ""
    analysis_id: str         = ""
    source_type: AudioSource = AudioSource.YOUTUBE
    dest_dir:    Path        = Path(".")

    @classmethod
    def from_url(cls, source_url: str, analysis_id: str, dest_dir: Path) -> AudioFile:
        """Factory — infer source_type from URL."""
        source_type = (
            AudioSource.YOUTUBE if source_url.startswith("http")
            else AudioSource.LOCAL
        )
        return cls(
            source_url=source_url,
            analysis_id=analysis_id,
            source_type=source_type,
            dest_dir=dest_dir,
        )
