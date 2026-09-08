"""YoutubeDownloader — Downloader implementation for YouTube URLs.

Wraps the existing DownloadHandler logic exactly as-is.
Only the import paths and the interface (AudioFile → DownloadResponse) are new.
"""
from __future__ import annotations
from pathlib import Path

from src.data_loader.domain.models.audio_file import AudioFile
from src.data_loader.domain.models.download_response import DownloadResponse


class YoutubeDownloader:
    """Downloader: YouTube URL → WAV 44100Hz mono.

    Logic is identical to the original DownloadHandler._download_youtube().
    Wraps it to conform to the Downloader Protocol interface.
    """

    # ffmpeg args: -ar 44100 (resample), -ac 1 (mono)
    _YDL_OPTS = {
        "format": "bestaudio/best",
        "outtmpl": "%(id)s.%(ext)s",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
        "postprocessor_args": {
            "ffmpeg": ["-ar", "44100", "-ac", "1"]
        },
        "quiet": True,
        "no_warnings": True,
    }

    def download(self, audio_file: AudioFile) -> DownloadResponse:
        """YouTube URL → WAV 44100Hz mono saved to audio_file.dest_dir."""
        import yt_dlp  # lazy import — only available inside Docker container
        out_dir = audio_file.dest_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        opts = {**self._YDL_OPTS, "outtmpl": str(out_dir / "%(id)s.%(ext)s")}

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(audio_file.source_url, download=True)
                video_id = info["id"]
                wav_path = out_dir / f"{video_id}.wav"

                if not wav_path.exists():
                    return DownloadResponse.err(audio_file, f"WAV not found after download: {wav_path}")

                return DownloadResponse.ok(audio_file, wav_path)

        except Exception as e:
            return DownloadResponse.err(audio_file, f"Download failed: {e}")
