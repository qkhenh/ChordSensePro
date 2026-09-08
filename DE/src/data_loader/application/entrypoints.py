"""data_loader entrypoint — single public function called by data_ingest."""
from __future__ import annotations

from src.data_loader.domain.models.audio_file import AudioFile, AudioSource
from src.data_loader.domain.models.download_response import DownloadResponse
from src.data_loader.domain.services.audio_dispatcher import AudioDispatcher
from src.data_loader.application.downloaders.youtube_downloader import YoutubeDownloader
from src.data_loader.application.downloaders.local_file_downloader import LocalFileDownloader


def run_data_loader(audio_file: AudioFile) -> DownloadResponse:
    """Entry point for the E (Extract) phase.

    Wires the AudioDispatcher registry and dispatches the download.
    Called exclusively by data_ingest.IngestService — not by DAGs directly.

    Args:
        audio_file: AudioFile containing source_url, dest_dir, source_type.

    Returns:
        DownloadResponse with wav_path on success, error on failure.
    """
    dispatcher = AudioDispatcher()
    dispatcher.register(AudioSource.YOUTUBE, YoutubeDownloader())
    dispatcher.register(AudioSource.LOCAL,   LocalFileDownloader())
    return dispatcher.download(audio_file)
