"""JaahDownloader — downloads JAAH jazz dataset tracks from Zenodo.

JAAH Dataset:
  - 113 jazz tracks with complete dim7/maj7/aug chord annotations
  - License: CC BY 4.0 (Zenodo)
  - URL: https://zenodo.org/record/1290737
  - Format: MP3 + JSON annotations
  - Purpose: Head 2+3 training (Quality + Extension)

Reference: Yurchenko, K. et al. "JAAH: Audio-Aligned Jazz Harmony Dataset"
"""
from __future__ import annotations
import urllib.request
from pathlib import Path

from src.data_loader.domain.models.audio_file import AudioFile
from src.data_loader.domain.models.download_response import DownloadResponse


class JaahDownloader:
    """Downloader for JAAH dataset — direct HTTP download from Zenodo."""

    def download(self, audio_file: AudioFile) -> DownloadResponse:
        """Download JAAH track from Zenodo URL, convert to WAV 44100Hz."""
        out_dir = audio_file.dest_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # Derive filename from URL
        filename = audio_file.source_url.split("/")[-1]
        tmp_path  = out_dir / filename

        try:
            urllib.request.urlretrieve(audio_file.source_url, tmp_path)
            wav_path = self._to_wav(tmp_path, out_dir)
            tmp_path.unlink(missing_ok=True)   # remove intermediate MP3
            return DownloadResponse.ok(audio_file, wav_path)
        except Exception as e:
            return DownloadResponse.err(audio_file, f"JAAH download failed: {e}")

    def _to_wav(self, src: Path, out_dir: Path) -> Path:
        """Convert MP3 → WAV 44100Hz mono via librosa + soundfile."""
        import librosa
        import soundfile as sf
        audio, _ = librosa.load(str(src), sr=44100, mono=True)
        out_path = out_dir / (src.stem + ".wav")
        sf.write(str(out_path), audio, 44100)
        return out_path
