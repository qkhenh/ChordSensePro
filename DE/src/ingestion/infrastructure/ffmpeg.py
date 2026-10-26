"""ffprobe/ffmpeg adapter of the Normalizer port (subprocess, no shell)."""

import asyncio
import hashlib
import json
import wave
from pathlib import Path

from src.analysis_contract.models import ErrorCode
from src.ingestion.application.normalizer import (
    CANONICAL_SAMPLE_RATE,
    MediaInfo,
    NormalizedAudio,
)
from src.ingestion.domain.errors import IngestionError
from src.ingestion.domain.naming import safe_stem


def _unsupported(reason: str) -> IngestionError:
    return IngestionError(ErrorCode.UNSUPPORTED_FORMAT, reason)


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _wav_duration_ms(path: Path) -> int:
    with wave.open(str(path)) as wav:
        return round(wav.getnframes() * 1000 / wav.getframerate())


def _discard(path: Path) -> None:
    path.unlink(missing_ok=True)


class FfmpegNormalizer:
    def __init__(
        self, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe", timeout_s: float = 120
    ) -> None:
        self._ffmpeg = ffmpeg
        self._ffprobe = ffprobe
        self._timeout_s = timeout_s

    async def _run(self, program: str, *args: str) -> tuple[int, bytes]:
        try:
            process = await asyncio.create_subprocess_exec(
                program,
                *args,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"{program} was not found: install ffmpeg and put it on PATH"
            ) from None
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), self._timeout_s)
        except TimeoutError:
            process.kill()
            await process.wait()
            raise _unsupported("the audio could not be processed in time") from None
        return process.returncode or 0, stdout

    async def probe(self, path: Path) -> MediaInfo:
        code, stdout = await self._run(
            self._ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=format_name,duration:stream=codec_type",
            "-of",
            "json",
            str(path),
        )
        if code != 0:
            raise _unsupported("the file is not readable audio")
        try:
            data = json.loads(stdout)
            streams = data.get("streams", [])
            fmt = data["format"]
        except (ValueError, KeyError):
            raise _unsupported("the file is not readable audio") from None
        if not any(s.get("codec_type") == "audio" for s in streams):
            raise _unsupported("the file has no audio stream")
        try:
            duration_ms: int | None = round(float(fmt["duration"]) * 1000)
        except (KeyError, TypeError, ValueError):
            duration_ms = None
        return MediaInfo(format_name=str(fmt.get("format_name", "")), duration_ms=duration_ms)

    async def normalize(self, src: Path, dest_dir: Path, stem: str) -> NormalizedAudio:
        dest = dest_dir / f"{safe_stem(stem)}.wav"
        try:
            code, _ = await self._run(
                self._ffmpeg,
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-i",
                str(src),
                "-vn",
                "-map_metadata",
                "-1",
                "-fflags",
                "+bitexact",
                "-flags:a",
                "+bitexact",
                "-ar",
                str(CANONICAL_SAMPLE_RATE),
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(dest),
            )
            if code != 0 or not dest.exists():
                raise _unsupported("the audio could not be decoded")
            duration_ms = await asyncio.to_thread(_wav_duration_ms, dest)
            if duration_ms <= 0:
                raise _unsupported("the audio is empty")
            return NormalizedAudio(dest, await asyncio.to_thread(_sha256_of, dest), duration_ms)
        except BaseException:
            _discard(dest)
            raise
