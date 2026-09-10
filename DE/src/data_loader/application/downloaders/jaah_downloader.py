"""JaahDownloader — downloads JAAH jazz dataset tracks via YouTube search.

JAAH Dataset:
  - 113 jazz tracks with complete dim7/maj7/aug chord annotations
  - License: CC BY 4.0 (annotations) | Commercial recordings (audio)
  - Annotations: https://github.com/MTG/JAAH (local after clone)
  - Audio: NOT redistributed — downloads via YouTube search (artist + title)
  - source_url format: "ytsearch1:{artist} {title} jazz"

Reference: Yurchenko, K. et al. "JAAH: Audio-Aligned Jazz Harmony Dataset"
"""
from __future__ import annotations
from pathlib import Path

from src.data_loader.domain.models.audio_file import AudioFile
from src.data_loader.domain.models.download_response import DownloadResponse


class JaahDownloader:
    """Downloader for JAAH dataset — finds audio via YouTube search + yt-dlp.

    source_url is a yt-dlp search query: "ytsearch1:{artist} {title} jazz"
    Falls back to direct URL if source_url starts with https://.
    """

    _YDL_OPTS = {
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
        "postprocessor_args": {
            "ffmpeg": ["-ar", "44100", "-ac", "1"]
        },
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }

    def download(self, audio_file: AudioFile) -> DownloadResponse:
        """Download JAAH track via YouTube search or direct URL."""
        import yt_dlp

        out_dir = audio_file.dest_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        opts = {
            **self._YDL_OPTS,
            "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        }

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                # source_url is either "ytsearch1:..." or a direct https:// URL
                info = ydl.extract_info(audio_file.source_url, download=True)

                # ytsearch returns a playlist wrapper — get first entry
                if "entries" in info:
                    info = info["entries"][0]

                wav_path = out_dir / f"{info['id']}.wav"
                if not wav_path.exists():
                    return DownloadResponse.err(audio_file, f"WAV not found: {wav_path}")

                return DownloadResponse.ok(audio_file, wav_path)

        except Exception as e:
            return DownloadResponse.err(audio_file, f"JAAH download failed: {e}")
