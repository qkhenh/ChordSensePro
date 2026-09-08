"""RwcDownloader — downloads RWC Popular Music dataset tracks.

RWC Popular Music Database:
  - 100 tracks with beat-level chord annotation
  - License: Free (requires registration at https://staff.aist.go.jp/m.goto/RWC-MDB/)
  - Audio format: WAV or AIFF, redistributed via institutional access
  - After registration, files are provided as a download archive

Usage:
    source_url = "https://staff.aist.go.jp/m.goto/RWC-MDB/..."
    OR source_url = "/path/to/rwc_archive/RM-P001.wav" (if already downloaded)
"""
from __future__ import annotations
from pathlib import Path
import shutil

from src.data_loader.domain.models.audio_file import AudioFile
from src.data_loader.domain.models.download_response import DownloadResponse


class RwcDownloader:
    """Downloader for RWC Popular Music dataset.

    RWC requires manual registration and download. This downloader
    handles the case where the archive is already on disk (most common).
    Resamples to 44100Hz mono for pipeline compatibility.
    """

    def download(self, audio_file: AudioFile) -> DownloadResponse:
        """Copy + resample RWC audio file to dest_dir."""
        src = Path(audio_file.source_url)
        if not src.exists():
            return DownloadResponse.err(
                audio_file,
                f"RWC file not found: {src}. "
                "Download the RWC archive manually from https://staff.aist.go.jp/m.goto/RWC-MDB/"
            )
        return self._resample(audio_file, src)

    def _resample(self, audio_file: AudioFile, src: Path) -> DownloadResponse:
        """Resample to 44100Hz mono via librosa + soundfile."""
        try:
            import librosa
            import soundfile as sf
            out_dir = audio_file.dest_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            audio, _ = librosa.load(str(src), sr=44100, mono=True)
            out_path = out_dir / (src.stem + ".wav")
            sf.write(str(out_path), audio, 44100)
            return DownloadResponse.ok(audio_file, out_path)
        except Exception as e:
            return DownloadResponse.err(audio_file, f"RWC resample failed: {e}")
