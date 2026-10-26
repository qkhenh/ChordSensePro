"""Media resolvers and the mapping from a link to how the client plays it."""

import re
import shutil
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from src.analysis_contract.models import ErrorCode, PlaybackSource
from src.ingestion.domain.errors import IngestionError
from src.ingestion.domain.naming import safe_stem, safe_title
from src.ingestion.domain.ports import ResolvedMedia

_YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}
_SOUNDCLOUD_HOSTS = {
    "soundcloud.com",
    "www.soundcloud.com",
    "m.soundcloud.com",
    "on.soundcloud.com",
}
_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}")
_VIDEO_PATH = re.compile(r"^/(?:embed|shorts|live|v)/([A-Za-z0-9_-]{11})(?:/|$)")


def _youtube_id(host: str, path: str, query: str) -> str | None:
    if host == "youtu.be":
        candidate = path.strip("/").split("/")[0]
    elif path == "/watch":
        candidate = (parse_qs(query).get("v") or [""])[0]
    else:
        match = _VIDEO_PATH.match(path)
        candidate = match.group(1) if match else ""
    return candidate if _VIDEO_ID.fullmatch(candidate) else None


def playback_for_url(url: str) -> PlaybackSource:
    """YouTube/SoundCloud play in their embedded players (the server keeps no audio);
    anything else is a direct link the browser plays from its origin."""
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if host in _YOUTUBE_HOSTS or host == "youtu.be":
        video = _youtube_id(host, parts.path, parts.query)
        if video is None:
            raise IngestionError(ErrorCode.URL_UNRESOLVABLE, "the link has no video id")
        return PlaybackSource(kind="youtube", embed_id=video)
    if host in _SOUNDCLOUD_HOSTS:
        return PlaybackSource(
            kind="soundcloud", embed_id=f"{parts.scheme}://{parts.netloc}{parts.path}"
        )
    return PlaybackSource(kind="direct", audio_url=url)


class FakeMediaResolver:
    """M2 stand-in for the yt-dlp resolver: always "downloads" the bundled demo audio."""

    def __init__(self, demo_audio: Path) -> None:
        self._demo_audio = demo_audio

    async def resolve(self, url: str, dest_dir: Path) -> ResolvedMedia:
        if "fail-unresolvable" in url:
            raise IngestionError(ErrorCode.URL_UNRESOLVABLE, "the link cannot be resolved")
        parts = urlsplit(url)
        last = unquote(parts.path.rstrip("/").rsplit("/", 1)[-1])
        title = safe_title(last) if last else (parts.hostname or "Untitled")
        if title == "Untitled" and parts.hostname:
            title = parts.hostname
        dest = dest_dir / f"{safe_stem(last or title)}.wav"
        shutil.copyfile(self._demo_audio, dest)
        return ResolvedMedia(path=dest, title=title, artist=None, playback=playback_for_url(url))
