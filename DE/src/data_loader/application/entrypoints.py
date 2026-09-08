"""data_loader entrypoint — single public function called by data_ingest."""
from __future__ import annotations

from src.data_loader.domain.models.audio_file import AudioFile, AudioSource
from src.data_loader.domain.models.download_response import DownloadResponse
from src.data_loader.domain.services.audio_dispatcher import AudioDispatcher
from src.data_loader.application.downloaders.rwc_downloader import RwcDownloader
from src.data_loader.application.downloaders.mcgill_downloader import McGillDownloader
from src.data_loader.application.downloaders.jaah_downloader import JaahDownloader
from src.data_loader.application.downloaders.choco_downloader import ChocoDownloader
from src.data_loader.application.downloaders.pianoteq_downloader import PianoteqDownloader
from src.data_loader.application.downloaders.kaggle_downloader import KaggleDownloader
from src.data_loader.application.downloaders.local_file_downloader import LocalFileDownloader


def _build_dispatcher() -> AudioDispatcher:
    """Wire all dataset-source Downloaders into the AudioDispatcher registry."""
    dispatcher = AudioDispatcher()
    dispatcher.register(AudioSource.RWC,      RwcDownloader())
    dispatcher.register(AudioSource.MCGILL,   McGillDownloader())
    dispatcher.register(AudioSource.JAAH,     JaahDownloader())
    dispatcher.register(AudioSource.CHOCO,    ChocoDownloader())
    dispatcher.register(AudioSource.PIANOTEQ, PianoteqDownloader())
    dispatcher.register(AudioSource.KAGGLE,   KaggleDownloader())
    dispatcher.register(AudioSource.LOCAL,    LocalFileDownloader())
    return dispatcher


def run_data_loader(audio_file: AudioFile) -> DownloadResponse:
    """Entry point for the E (Extract) phase.

    Dispatches download to the correct Downloader based on audio_file.source_type.
    Called exclusively by data_ingest.IngestService — not by DAGs directly.

    Dataset sources (from 18_DE_AI_chordsense.md section 2.2):
        RWC      — 100 tracks, beat chord annotation (manual archive download)
        MCGILL   — 1300 tracks, JAMS format (yt-dlp via YouTube IDs)
        JAAH     — 113 jazz tracks (direct Zenodo HTTP download)
        CHOCO    — 20,000 tracks, Harte notation (yt-dlp via YouTube IDs)
        PIANOTEQ — Synthetic render of extended chords (Pianoteq CLI)
        KAGGLE   — Guitar v3, 7,000+ recordings (Kaggle API CLI)
        LOCAL    — Local file already on disk (testing / manual additions)

    Args:
        audio_file: AudioFile with source_url, dest_dir, source_type set.

    Returns:
        DownloadResponse with wav_path on success, error on failure.
    """
    dispatcher = _build_dispatcher()
    return dispatcher.download(audio_file)
