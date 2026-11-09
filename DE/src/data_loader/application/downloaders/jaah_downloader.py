"""JaahDownloader — downloads JAAH jazz dataset tracks via YouTube search.

JAAH Dataset:
  - 113 jazz tracks with complete dim7/maj7/aug chord annotations
  - License: CC BY 4.0 (annotations) | Commercial recordings (audio)
  - Annotations: https://github.com/MTG/JAAH (local after clone)
  - Audio: NOT redistributed — downloads via YouTube search (artist + title)
  - source_url format: "ytsearch1:{artist} {title} jazz"

Reference: Yurchenko, K. et al. "JAAH: Audio-Aligned Jazz Harmony Dataset"

Candidate selection (added): the first search hit is often a live take / remaster / compilation
whose length differs from the annotated CD rip, which silently shifts every chord label. When the
annotation's ``duration`` is known (AudioFile.expected_duration) we search several results and
download the one whose duration matches (see ``_ytdlp.pick_best_candidate``). If nothing matches
within tolerance the track fails fast, before the expensive Demucs step.
"""
from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Any

from src.data_loader.application.downloaders._ytdlp import (
    alignment_tolerance, base_opts, pick_best_candidate,
)
from src.data_loader.domain.models.audio_file import AudioFile
from src.data_loader.domain.models.download_response import DownloadResponse


class JaahDownloader:
    """Downloader for JAAH dataset — finds audio via YouTube search + yt-dlp.

    source_url is a yt-dlp search query: "ytsearch1:{artist} {title} jazz"
    Falls back to direct URL if source_url starts with https://.
    """

    def download(self, audio_file: AudioFile) -> DownloadResponse:
        """Download JAAH track via YouTube search or direct URL."""
        import yt_dlp

        out_dir = audio_file.dest_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        target = audio_file.source_url

        try:
            if target.startswith("ytsearch") and audio_file.expected_duration is not None:
                target, error = self._resolve_candidate(target, audio_file.expected_duration)
                if error:
                    return DownloadResponse.err(audio_file, error)

            with yt_dlp.YoutubeDL(base_opts(out_dir)) as ydl:
                info = ydl.extract_info(target, download=True)

                # ytsearch returns a playlist wrapper — get first entry
                if info is not None and "entries" in info:
                    entries = [e for e in info["entries"] if e]
                    if not entries:
                        return DownloadResponse.err(audio_file, f"YouTube search returned no results: {target}")
                    info = entries[0]
                if not info:
                    return DownloadResponse.err(audio_file, f"yt-dlp returned no info: {target}")

                wav_path = out_dir / f"{info['id']}.wav"
                if not wav_path.exists():
                    return DownloadResponse.err(audio_file, f"WAV not found: {wav_path}")

                return DownloadResponse.ok(audio_file, wav_path)

        except Exception as e:
            return DownloadResponse.err(audio_file, f"JAAH download failed: {e}")

    @staticmethod
    def _resolve_candidate(search_url: str, expected_duration: float) -> tuple[str, str | None]:
        """Search several results (no download) and return (watch_url, None) or ('', error)."""
        import yt_dlp

        n_candidates = int(os.getenv("YTDLP_SEARCH_CANDIDATES", "10"))
        query = re.sub(r"^ytsearch\d*:", "", search_url)

        opts = base_opts(Path("."))
        flat_opts: Any = {
            "quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True,
            "socket_timeout": opts["socket_timeout"], "extractor_retries": opts["extractor_retries"],
        }
        if "cookiefile" in opts:
            flat_opts["cookiefile"] = opts["cookiefile"]

        with yt_dlp.YoutubeDL(flat_opts) as ydl:
            result = ydl.extract_info(f"ytsearch{n_candidates}:{query}", download=False)

        entries = [e for e in (result or {}).get("entries", []) if e]
        if not entries:
            return "", f"YouTube search returned no results: {query}"

        # Flat results should carry a duration; if none do, we cannot choose → keep the top hit
        # and rely on AlignmentCheckHandler to reject a wrong version later.
        if not any(e.get("duration") for e in entries):
            best = entries[0]
        else:
            tolerance = alignment_tolerance()
            best = pick_best_candidate(entries, expected_duration, tolerance)
            if best is None:
                durations = sorted({round(float(d)) for d in (e.get("duration") for e in entries) if d})
                return "", (
                    f"No YouTube result within {tolerance:.0f}s of annotated duration "
                    f"{expected_duration:.0f}s (candidates: {durations[:8]})"
                )
        return f"https://www.youtube.com/watch?v={best['id']}", None
