"""Tiny audio fixtures for ingestion tests (stdlib ``wave`` + the real ffmpeg when needed)."""

import math
import shutil
import struct
import subprocess
import wave
from pathlib import Path

import pytest

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)


def wav_bytes_header_only(seconds: float, rate: int = 8000) -> bytes:
    """A syntactically valid WAV whose header declares ``seconds`` of mono 16-bit audio."""
    frames = int(seconds * rate)
    data_len = frames * 2
    return (
        b"RIFF"
        + struct.pack("<I", 36 + data_len)
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", data_len)
    )


def write_wav(
    path: Path, seconds: float = 1.0, rate: int = 22050, channels: int = 2, freq: float = 440.0
) -> Path:
    frames = bytearray()
    for i in range(int(seconds * rate)):
        sample = int(12000 * math.sin(2 * math.pi * freq * i / rate))
        frames += struct.pack("<h", sample) * channels
    with wave.open(str(path), "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(bytes(frames))
    return path


def encode(src: Path, dest: Path) -> Path:
    """Transcode with ffmpeg; the container/codec follow the destination extension."""
    subprocess.run(  # noqa: S603 - test helper, fixed arguments
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), str(dest)],  # noqa: S607
        check=True,
    )
    return dest
