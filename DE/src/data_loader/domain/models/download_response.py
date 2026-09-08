"""DownloadResponse — output model from any Downloader implementation."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from src.data_loader.domain.models.audio_file import AudioFile


class DownloadStatus(str, Enum):
    SUCCESS = "success"
    FAILED  = "failed"


@dataclass
class DownloadResponse:
    """Standard output from any Downloader — passed to data_processing."""
    audio_file:   AudioFile
    status:       DownloadStatus
    wav_path:     Path | None = None
    error:        str | None  = None
    file_size_mb: float | None = None
    duration_s:   float | None = None

    @property
    def is_ok(self) -> bool:
        return self.status == DownloadStatus.SUCCESS

    @property
    def is_err(self) -> bool:
        return self.status == DownloadStatus.FAILED

    @classmethod
    def ok(cls, audio_file: AudioFile, wav_path: Path, **kwargs) -> DownloadResponse:
        return cls(audio_file=audio_file, status=DownloadStatus.SUCCESS,
                   wav_path=wav_path, **kwargs)

    @classmethod
    def err(cls, audio_file: AudioFile, error: str) -> DownloadResponse:
        return cls(audio_file=audio_file, status=DownloadStatus.FAILED, error=error)
