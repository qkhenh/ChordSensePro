"""Shared yt-dlp configuration + candidate selection for the YouTube-backed downloaders.

Why this exists
---------------
JAAH and ChoCo both fetch audio from YouTube. Their option dicts were duplicated and had no
retry / timeout settings, so a single transient hiccup (HTTP 403, connection reset) marked a
URL as permanently failed. Everything that is common lives here:

  * ``base_opts``           retries, socket timeout, polite sleep, optional cookies file
  * ``pick_best_candidate`` choose the search result whose duration matches the annotation

YouTube note: recent yt-dlp releases need a JS runtime (deno) + the ``yt-dlp-ejs`` solver
scripts to extract YouTube formats at all — both are installed via requirements.txt.
Age-gated / "confirm you're not a bot" videos need a cookies.txt (env YTDLP_COOKIES).
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Sequence


def base_opts(out_dir: Path) -> Any:
    """yt-dlp options shared by every YouTube downloader (download → mono 44.1 kHz WAV)."""
    opts: dict[str, Any] = {
        "format": "bestaudio/best",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "postprocessor_args": {"ffmpeg": ["-ar", "44100", "-ac", "1"]},
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        # resilience: transient network/CDN errors are retried instead of failing the URL
        "retries": 5,
        "fragment_retries": 5,
        "extractor_retries": 3,
        "socket_timeout": 30,
        # be polite — hammering YouTube is the fastest way to get 403 / rate limited
        "sleep_interval": 1,
        "max_sleep_interval": 3,
    }
    cookies = os.getenv("YTDLP_COOKIES", "").strip()
    if cookies and Path(cookies).is_file():
        opts["cookiefile"] = cookies
    return opts


def pick_best_candidate(
    entries: Sequence[Any],
    expected_duration: float | None,
    tolerance: float,
) -> dict | None:
    """Pick the search result whose duration is closest to ``expected_duration``.

    * ``expected_duration`` None → no information, take the first valid entry (old behaviour).
    * Entries without a duration are ignored when a duration is expected.
    * Returns None when nothing is within ``tolerance`` seconds — the caller should fail fast
      (cheaper than downloading + running Demucs on a recording that cannot be aligned).
    """
    valid = [e for e in entries if e and e.get("id")]
    if not valid:
        return None
    if expected_duration is None:
        return valid[0]

    best: dict | None = None
    best_diff = float("inf")
    for entry in valid:
        duration = entry.get("duration")
        if duration is None:
            continue
        diff = abs(float(duration) - expected_duration)
        if diff < best_diff:
            best, best_diff = entry, diff
    return best if best is not None and best_diff <= tolerance else None


def alignment_tolerance() -> float:
    """Duration tolerance in seconds — same env var as AlignmentCheckHandler."""
    return float(os.getenv("ALIGN_TOLERANCE_SEC", "3.0"))
