"""AudioDispatcher — Registry pattern: maps AudioSource → Downloader implementation."""
from __future__ import annotations

from src.data_loader.domain.models.audio_file import AudioFile, AudioSource
from src.data_loader.domain.models.download_response import DownloadResponse
from src.data_loader.domain.ports.downloader import Downloader


class AudioDispatcher:
    """Registry dispatcher — maps AudioSource enum → Downloader.

    Usage:
        dispatcher = AudioDispatcher()
        dispatcher.register(AudioSource.YOUTUBE, YoutubeDownloader())
        dispatcher.register(AudioSource.LOCAL,   LocalFileDownloader())
        response = dispatcher.download(audio_file)

    Adding a new source: create Downloader + register(). No other changes needed.
    """

    def __init__(self) -> None:
        self._registry: dict[AudioSource, Downloader] = {}

    def register(self, source: AudioSource, downloader: Downloader) -> None:
        """Register a Downloader for a given AudioSource."""
        self._registry[source] = downloader

    def download(self, audio_file: AudioFile) -> DownloadResponse:
        """Dispatch download to the registered Downloader for audio_file.source_type."""
        downloader = self._registry.get(audio_file.source_type)
        if downloader is None:
            return DownloadResponse.err(
                audio_file,
                f"No downloader registered for source: {audio_file.source_type}"
            )
        return downloader.download(audio_file)
