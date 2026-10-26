"""Cheap, synchronous checks on an upload (NFR-12): size, duration, magic bytes."""

from typing import TYPE_CHECKING

from src.analysis_contract.models import ErrorCode
from src.ingestion.domain.errors import IngestionError
from src.ingestion.settings import IngestSettings

if TYPE_CHECKING:
    from src.ingestion.application.normalizer import MediaInfo

ALLOWED_FORMATS = ("mp3", "wav", "flac", "m4a", "ogg")


def _unsupported(reason: str) -> IngestionError:
    return IngestionError(ErrorCode.UNSUPPORTED_FORMAT, reason)


def sniff_format(head: bytes) -> str | None:
    """Name the audio format from the first bytes, or None when it is not one of the five."""
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "wav"
    if head[:4] == b"fLaC":
        return "flac"
    if head[:4] == b"OggS":
        return "ogg"
    if head[4:8] == b"ftyp":
        return "m4a"
    if head[:3] == b"ID3" or (len(head) >= 2 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0):
        return "mp3"
    return None


def check_limits(size_bytes: int, duration_ms: int | None, *, settings: IngestSettings) -> None:
    if size_bytes <= 0:
        raise _unsupported("the file is empty")
    if size_bytes > settings.max_upload_bytes:
        raise IngestionError(ErrorCode.FILE_TOO_LARGE, "the file is larger than the 100 MB limit")
    if duration_ms is None or duration_ms <= 0:
        raise _unsupported("the audio has no readable duration")
    if duration_ms > settings.max_duration_ms:
        raise IngestionError(ErrorCode.DURATION_TOO_LONG, "the audio is longer than 10 minutes")


def check_format(head: bytes, info: "MediaInfo") -> None:
    """The magic bytes must name one of the five formats and ffprobe must agree."""
    sniffed = sniff_format(head)
    if sniffed is None:
        raise _unsupported("the file is not an MP3, WAV, FLAC, M4A or OGG audio file")
    if sniffed not in info.format_name.split(","):
        raise _unsupported("the file content does not match its audio format")
