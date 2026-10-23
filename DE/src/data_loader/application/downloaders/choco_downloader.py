"""ChocoDownloader — handles ChoCo Corpus audio files.

ChoCo Corpus:
  - 20,000 tracks from 18 sources, Harte notation
  - License: CC BY 4.0 (GitHub: https://github.com/smashub/choco)
  - Audio: NOT included in repo — sourced from YouTube IDs embedded in annotations
  - Annotations: Harte notation → parsed separately in T phase

Note: ChoCo is primarily an annotation corpus. Audio must be fetched
via YouTube IDs embedded in each track's metadata JSON.
source_url = YouTube URL extracted from ChoCo track metadata (chordify/CASD partition), or
'ytsearch1:<performer> <title>' for the audio partitions of the ChoCo release (Isophonics,
USpop2002, RWC-Pop, Robbie Williams, Billboard) that only carry title/performer/duration. In the
search case the candidate whose length matches the JAMS ``file_metadata.duration`` is downloaded
(same logic as JAAH, see jaah_downloader.py).
"""
from __future__ import annotations

from src.data_loader.application.downloaders._ytdlp import base_opts
from src.data_loader.domain.models.audio_file import AudioFile
from src.data_loader.domain.models.download_response import DownloadResponse


class ChocoDownloader:
    """Downloader for ChoCo Corpus — audio via yt-dlp (YouTube source).

    ChoCo provides annotations only; audio is fetched via embedded YouTube IDs.
    Functionally identical to McGillDownloader — separate class for clarity and
    to allow future customization (e.g. different retry logic, rate limiting).
    """

    def download(self, audio_file: AudioFile) -> DownloadResponse:
        """Download ChoCo track via yt-dlp using YouTube ID from annotation."""
        if audio_file.source_url.startswith("ytsearch"):
            from src.data_loader.application.downloaders.jaah_downloader import JaahDownloader
            response = JaahDownloader().download(audio_file)
            if response.is_err and response.error:
                response.error = response.error.replace("JAAH download failed", "ChoCo search download failed")
            return response

        import yt_dlp  # lazy — Docker only
        out_dir = audio_file.dest_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        opts = base_opts(out_dir)   # shared: retries, timeouts, optional cookies
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(audio_file.source_url, download=True)
                wav_path = out_dir / f"{info['id']}.wav"
                if not wav_path.exists():
                    return DownloadResponse.err(audio_file, f"WAV not found: {wav_path}")
                return DownloadResponse.ok(audio_file, wav_path)
        except Exception as e:
            return DownloadResponse.err(audio_file, f"ChoCo download failed: {e}")
