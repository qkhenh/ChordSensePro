"""AudioSource enum — supported academic dataset sources for ETL training data ingest.

Based on 18_DE_AI_chordsense.md section 2.2 Dataset Strategy.
Each source has a dedicated Downloader implementation registered in AudioDispatcher.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from src.shared.domain.base_model import BaseEntity


class AudioSource(str, Enum):
    """Academic dataset sources for training data ETL.

    Sources per MD spec (section 2.2):
        RWC       — RWC Popular Music, 100 tracks, beat-level chord annotation
        MCGILL    — McGill Billboard 1300, JAMS format, CC Research license
        JAAH      — 113 jazz tracks with dim7/maj7/aug, CC (Zenodo)
        CHOCO     — ChoCo Corpus 20,000 tracks, Harte notation, CC (GitHub)
        PIANOTEQ  — Synthetic render of extended chords via Pianoteq (internal)
        KAGGLE    — Kaggle Guitar v3, 7,000+ isolated chord recordings (optional)
        LOCAL     — Local file already on disk (for testing / manual additions)
    """
    RWC      = "rwc"
    MCGILL   = "mcgill"
    JAAH     = "jaah"
    CHOCO    = "choco"
    PIANOTEQ = "pianoteq"
    KAGGLE   = "kaggle"
    LOCAL    = "local"


@dataclass
class AudioFile(BaseEntity):
    """Represents one audio file to be downloaded and processed.

    Created by data_ingest, passed to AudioDispatcher.
    source_type drives which Downloader implementation is used.

    For dataset sources (RWC, McGill, JAAH, ChoCo, Kaggle):
        source_url is the remote URL or dataset identifier.
    For PIANOTEQ:
        source_url is the chord label to render (e.g. 'Cmaj7#11').
    For LOCAL:
        source_url is the absolute local file path.
    """
    source_url:   str         = ""
    analysis_id:  str         = ""
    source_type:  AudioSource = AudioSource.LOCAL
    dest_dir:     Path        = Path(".")
    dataset_name: str         = ""   # human-readable label for logging/tracking

    @classmethod
    def from_url(cls, source_url: str, analysis_id: str, dest_dir: Path,
                 source_type: AudioSource | None = None,
                 dataset_name: str = "") -> AudioFile:
        """Factory — infer source_type from URL if not explicitly provided."""
        if source_type is None:
            # Best-effort inference — callers should pass source_type explicitly
            source_type = AudioSource.LOCAL
        return cls(
            source_url=source_url,
            analysis_id=analysis_id,
            source_type=source_type,
            dest_dir=dest_dir,
            dataset_name=dataset_name,
        )
