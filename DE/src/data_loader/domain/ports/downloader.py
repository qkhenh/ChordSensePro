"""Downloader Protocol — interface that all Downloader implementations must satisfy."""
from __future__ import annotations
from typing import Protocol

from src.data_loader.domain.models.audio_file import AudioFile
from src.data_loader.domain.models.download_response import DownloadResponse


class Downloader(Protocol):
    """Port: any class with this method signature is a valid Downloader.

    Implementations:
        YoutubeDownloader  — wraps yt-dlp via DownloadHandler
        LocalFileDownloader — copies local file to dest_dir
    """
    def download(self, audio_file: AudioFile) -> DownloadResponse: ...
