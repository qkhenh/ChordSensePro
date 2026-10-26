import hashlib
import wave

import pytest

from src.analysis_contract.models import ErrorCode
from src.ingestion.domain.errors import IngestionError
from src.ingestion.infrastructure.ffmpeg import FfmpegNormalizer
from tests.support.audio import encode, needs_ffmpeg, write_wav

pytestmark = needs_ffmpeg


@pytest.fixture
def normalizer():
    return FfmpegNormalizer()


async def code_of(awaitable) -> ErrorCode:
    with pytest.raises(IngestionError) as exc:
        await awaitable
    return exc.value.code


async def test_probe_reports_format_and_duration(normalizer, tmp_path):
    src = write_wav(tmp_path / "a.wav", seconds=1.5)
    info = await normalizer.probe(src)
    assert info.format_name == "wav"
    assert info.duration_ms == pytest.approx(1500, abs=20)


async def test_normalizing_yields_44k_mono_16_bit_wav(normalizer, tmp_path):
    src = write_wav(tmp_path / "in.wav", seconds=1.0, rate=22050, channels=2)
    out = await normalizer.normalize(src, tmp_path, "My Song")
    assert out.path == tmp_path / "my-song.wav"
    with wave.open(str(out.path)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (44100, 1, 2)
    assert out.duration_ms == pytest.approx(1000, abs=20)
    assert out.content_hash == hashlib.sha256(out.path.read_bytes()).hexdigest()


async def test_the_same_audio_always_gets_the_same_content_hash(normalizer, tmp_path):
    src = write_wav(tmp_path / "in.wav", seconds=1.0)
    (tmp_path / "x").mkdir()
    (tmp_path / "y").mkdir()
    first = await normalizer.normalize(src, tmp_path / "x", "a")
    second = await normalizer.normalize(src, tmp_path / "y", "b")
    assert first.content_hash == second.content_hash


async def test_lossless_re_encoding_keeps_the_hash(normalizer, tmp_path):
    wav = write_wav(tmp_path / "in.wav", seconds=1.0)
    flac = encode(wav, tmp_path / "in.flac")
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    from_wav = await normalizer.normalize(wav, tmp_path / "a", "s")
    from_flac = await normalizer.normalize(flac, tmp_path / "b", "s")
    assert from_wav.content_hash == from_flac.content_hash


@pytest.mark.parametrize("suffix", ["mp3", "ogg", "m4a", "flac"])
async def test_every_supported_container_is_probed_and_normalized(normalizer, tmp_path, suffix):
    src = encode(write_wav(tmp_path / "in.wav", seconds=1.0), tmp_path / f"in.{suffix}")
    assert (await normalizer.probe(src)).duration_ms == pytest.approx(1000, abs=120)
    out = await normalizer.normalize(src, tmp_path, "song")
    assert out.path.exists()


async def test_a_text_file_named_mp3_is_unsupported(normalizer, tmp_path):
    fake = tmp_path / "fake.mp3"
    fake.write_text("this is not audio at all")
    assert await code_of(normalizer.probe(fake)) == ErrorCode.UNSUPPORTED_FORMAT
    assert (
        await code_of(normalizer.normalize(fake, tmp_path, "fake")) == ErrorCode.UNSUPPORTED_FORMAT
    )
    assert not (tmp_path / "fake.wav").exists()


async def test_an_empty_file_is_unsupported(normalizer, tmp_path):
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    assert await code_of(normalizer.probe(empty)) == ErrorCode.UNSUPPORTED_FORMAT


async def test_a_file_without_an_audio_stream_is_unsupported(normalizer, tmp_path):
    image = tmp_path / "pic.png"
    image.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000d49444154789c6300010000000500010d0a2db40000000049454e44ae426082"
        )
    )
    assert await code_of(normalizer.probe(image)) == ErrorCode.UNSUPPORTED_FORMAT


async def test_a_decode_that_takes_too_long_is_cancelled_and_cleaned_up(tmp_path):
    src = write_wav(tmp_path / "in.wav", seconds=1.0)
    slow = FfmpegNormalizer(timeout_s=0.001)
    assert await code_of(slow.normalize(src, tmp_path, "slow")) == ErrorCode.UNSUPPORTED_FORMAT
    assert not (tmp_path / "slow.wav").exists()


async def test_a_missing_ffprobe_is_a_configuration_error_not_a_user_error(tmp_path):
    src = write_wav(tmp_path / "in.wav", seconds=0.1)
    with pytest.raises(RuntimeError, match="ffprobe"):
        await FfmpegNormalizer(ffprobe="definitely-not-installed-ffprobe").probe(src)
