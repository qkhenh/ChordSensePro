"""The Normalizer port: validate media and turn any source into the pipeline's canonical WAV."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

CANONICAL_SAMPLE_RATE = 44_100  # Hz, mono, 16-bit (markdown/ChordSense.MD)


@dataclass(frozen=True)
class MediaInfo:
    format_name: str  # ffprobe's container list, e.g. "mov,mp4,m4a,3gp,3g2,mj2"
    duration_ms: int | None


@dataclass(frozen=True)
class NormalizedAudio:
    path: Path
    content_hash: str  # sha256 of the canonical WAV bytes
    duration_ms: int


class Normalizer(Protocol):
    async def probe(self, path: Path) -> MediaInfo:
        """Inspect a file. Raises ``IngestionError(UNSUPPORTED_FORMAT)`` if it is not audio."""
        ...

    async def normalize(self, src: Path, dest_dir: Path, stem: str) -> NormalizedAudio:
        """Transcode to ``dest_dir/<safe_stem(stem)>.wav`` (44.1 kHz mono 16-bit)."""
        ...
