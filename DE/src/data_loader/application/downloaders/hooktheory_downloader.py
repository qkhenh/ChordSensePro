"""HooktheoryDownloader — downloads only the annotated section of a Hooktheory clip's YouTube video.

Hooktheory annotations (Sheet Sage dataset, CC BY-NC-SA 3.0) label a short ~20 s clip of a song and
carry the YouTube video id the annotator aligned against. Downloading the whole video and running
Demucs on it would waste minutes per item, so only ``audio_file.section`` (the labelled region
plus a few seconds of padding) is fetched with yt-dlp's ``download_ranges``.

source_url format : ``https://www.youtube.com/watch?v=<id>#ht=<hooktheory id>``
                    (the ``#ht=`` fragment keeps several clips of one video unique in the queue;
                    it is stripped before the request)
Version guard     : the annotation stores the video duration the annotator saw; a different
                    upload (edit / re-encode with another length) would shift every chord, so the
                    item fails when the durations differ by more than ALIGN_TOLERANCE_SEC.
The WAV starts at ``section[0]`` — the annotation timeline is already expressed in that clip time.
"""
from __future__ import annotations

from src.data_loader.application.downloaders._ytdlp import alignment_tolerance, base_opts
from src.data_loader.domain.models.audio_file import AudioFile
from src.data_loader.domain.models.download_response import DownloadResponse


class HooktheoryDownloader:
    """Downloader for Hooktheory clips — YouTube section download via yt-dlp."""

    def download(self, audio_file: AudioFile) -> DownloadResponse:
        import yt_dlp  # lazy — Docker only
        from yt_dlp.utils import download_range_func

        out_dir = audio_file.dest_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        url = audio_file.source_url.split("#", 1)[0]
        section = audio_file.section

        opts = base_opts(out_dir)
        if section is not None:
            ranges = [(float(section[0]), float(section[1]))]      # yt-dlp accepts float seconds
            opts["download_ranges"] = download_range_func([], ranges)  # type: ignore[arg-type]
            opts["force_keyframes_at_cuts"] = True      # cut exactly at the section start

        # section downloads are streamed by ffmpeg itself, so yt-dlp's `retries` do not apply:
        # a dropped connection surfaces as "ffmpeg exited with code 1" → retry once.
        last_error = ""
        for attempt in range(2):
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    if not info:
                        return DownloadResponse.err(audio_file, f"yt-dlp returned no info: {url}")

                    expected = audio_file.expected_duration
                    actual = info.get("duration")
                    if expected is not None and actual is not None:
                        diff = abs(float(actual) - expected)
                        if diff > alignment_tolerance():
                            return DownloadResponse.err(
                                audio_file,
                                f"duration_mismatch: video {float(actual):.1f}s vs annotated {expected:.1f}s "
                                f"(Δ{diff:.1f}s) — different YouTube upload",
                            )

                    wav_path = out_dir / f"{info['id']}.wav"
                    if not wav_path.exists():
                        candidates = sorted(out_dir.glob("*.wav"))   # section downloads may add a suffix
                        if not candidates:
                            return DownloadResponse.err(audio_file, f"WAV not found: {wav_path}")
                        wav_path = candidates[0]
                    return DownloadResponse.ok(audio_file, wav_path)
            except Exception as e:
                last_error = str(e)
                if attempt == 0 and "ffmpeg exited" in last_error:
                    continue
                break
        return DownloadResponse.err(audio_file, f"Hooktheory download failed: {last_error}")
