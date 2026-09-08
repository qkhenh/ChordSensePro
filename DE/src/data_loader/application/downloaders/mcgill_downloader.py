"""McGillDownloader — downloads McGill Billboard dataset tracks.

McGill Billboard Dataset:
  - 1,300 pop/rock tracks with JAMS chord annotations
  - License: CC Research (non-commercial)
  - Audio: NOT redistributed — must download via Zenodo + YouTube IDs
  - Annotations: https://github.com/jimir/billboard-scraping
  - Audio source: YouTube IDs provided in annotation files → yt-dlp download

Reference: Burgoyne et al. "An Expert Ground Truth Set for Audio Chord Recognition and Music Analysis"
"""
from __future__ import annotations

from src.data_loader.domain.models.audio_file import AudioFile
from src.data_loader.domain.models.download_response import DownloadResponse


class McGillDownloader:
    """Downloader for McGill Billboard dataset.

    McGill audio is not directly redistributed. This downloader uses yt-dlp
    to download audio via YouTube IDs extracted from the annotation files.
    source_url = YouTube URL extracted from JAMS annotation.
    """

    # ffmpeg args: -ar 44100 (resample), -ac 1 (mono)
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
    }

    def download(self, audio_file: AudioFile) -> DownloadResponse:
        """Download McGill track via yt-dlp (YouTube source)."""
        import yt_dlp  # lazy — Docker only
        out_dir = audio_file.dest_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        opts = {**self._YDL_OPTS, "outtmpl": str(out_dir / "%(id)s.%(ext)s")}
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(audio_file.source_url, download=True)
                wav_path = out_dir / f"{info['id']}.wav"
                if not wav_path.exists():
                    return DownloadResponse.err(audio_file, f"WAV not found: {wav_path}")
                return DownloadResponse.ok(audio_file, wav_path)
        except Exception as e:
            return DownloadResponse.err(audio_file, f"McGill download failed: {e}")
