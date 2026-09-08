"""LocalFileDownloader — Downloader implementation for local file paths.

Wraps the existing DownloadHandler._convert_local() logic exactly as-is.
"""
from __future__ import annotations
from pathlib import Path

from src.data_loader.domain.models.audio_file import AudioFile
from src.data_loader.domain.models.download_response import DownloadResponse


class LocalFileDownloader:
    """Downloader: local file path → WAV 44100Hz mono.

    Logic is identical to the original DownloadHandler._convert_local().
    """

    def download(self, audio_file: AudioFile) -> DownloadResponse:
        """Local file → WAV 44100Hz mono via librosa + soundfile."""
        src = Path(audio_file.source_url)
        out_dir = audio_file.dest_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        if not src.exists():
            return DownloadResponse.err(audio_file, f"File not found: {src}")

        try:
            import soundfile as sf
            import librosa
            audio, sr = librosa.load(str(src), sr=44100, mono=True)
            out_path = out_dir / (src.stem + ".wav")
            sf.write(str(out_path), audio, 44100)
            return DownloadResponse.ok(audio_file, out_path)

        except Exception as e:
            return DownloadResponse.err(audio_file, f"Convert failed: {e}")
