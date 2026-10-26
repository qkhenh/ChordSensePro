import pytest

from src.analysis_contract.models import ErrorCode
from src.ingestion.application.normalizer import MediaInfo
from src.ingestion.domain.errors import IngestionError
from src.ingestion.domain.naming import safe_title
from src.ingestion.domain.upload_rules import check_format, check_limits, sniff_format
from src.ingestion.settings import IngestSettings

LIMITS = IngestSettings()
MAX_BYTES = 104_857_600
MAX_MS = 600_000


def code_of(call) -> ErrorCode:
    with pytest.raises(IngestionError) as exc:
        call()
    return exc.value.code


class TestLimits:
    def test_exactly_100_mib_and_exactly_ten_minutes_are_accepted(self):
        check_limits(MAX_BYTES, MAX_MS, settings=LIMITS)

    def test_one_byte_over_is_file_too_large(self):
        assert code_of(lambda: check_limits(MAX_BYTES + 1, 1000, settings=LIMITS)) == (
            ErrorCode.FILE_TOO_LARGE
        )

    def test_one_millisecond_over_is_duration_too_long(self):
        assert code_of(lambda: check_limits(1000, MAX_MS + 1, settings=LIMITS)) == (
            ErrorCode.DURATION_TOO_LONG
        )

    @pytest.mark.parametrize("duration", [None, 0, -5])
    def test_unknown_or_empty_duration_is_an_unsupported_format(self, duration):
        assert code_of(lambda: check_limits(1000, duration, settings=LIMITS)) == (
            ErrorCode.UNSUPPORTED_FORMAT
        )

    def test_zero_bytes_is_an_unsupported_format(self):
        assert code_of(lambda: check_limits(0, 1000, settings=LIMITS)) == (
            ErrorCode.UNSUPPORTED_FORMAT
        )

    def test_limits_follow_the_settings(self):
        small = IngestSettings(max_upload_bytes=10, max_duration_ms=50)
        check_limits(10, 50, settings=small)
        assert code_of(lambda: check_limits(11, 50, settings=small)) == ErrorCode.FILE_TOO_LARGE


class TestSniff:
    @pytest.mark.parametrize(
        ("head", "expected"),
        [
            (b"ID3\x04\x00\x00\x00\x00\x00\x00", "mp3"),
            (b"\xff\xfb\x90\x00" + b"\x00" * 12, "mp3"),
            (b"RIFF\x24\x00\x00\x00WAVEfmt ", "wav"),
            (b"fLaC\x00\x00\x00\x22" + b"\x00" * 8, "flac"),
            (b"\x00\x00\x00\x20ftypM4A " + b"\x00" * 4, "m4a"),
            (b"OggS\x00\x02" + b"\x00" * 10, "ogg"),
            (b"RIFF\x24\x00\x00\x00AVI LIST", None),
            (b"hello world, not audio", None),
            (b"\xff", None),
            (b"", None),
        ],
    )
    def test_recognises_the_five_formats_by_their_magic_bytes(self, head, expected):
        assert sniff_format(head) == expected


class TestCheckFormat:
    def info(self, name: str) -> MediaInfo:
        return MediaInfo(format_name=name, duration_ms=1000)

    def test_accepts_when_ffprobe_agrees_with_the_magic_bytes(self):
        check_format(b"fLaC" + b"\x00" * 12, self.info("flac"))
        check_format(
            b"\x00\x00\x00\x20ftypM4A " + b"\x00" * 4, self.info("mov,mp4,m4a,3gp,3g2,mj2")
        )

    def test_rejects_unknown_magic_bytes(self):
        assert code_of(lambda: check_format(b"hello", self.info("mp3"))) == (
            ErrorCode.UNSUPPORTED_FORMAT
        )

    def test_rejects_a_disguised_container(self):
        # An Ogg header that ffprobe decodes as something else is not what the name claims.
        assert code_of(
            lambda: check_format(b"OggS" + b"\x00" * 12, self.info("matroska,webm"))
        ) == (ErrorCode.UNSUPPORTED_FORMAT)


class TestSafeTitle:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Autumn Leaves.mp3", "Autumn Leaves"),
            ("../../etc/passwd.mp3", "passwd"),
            ("C:\\music\\Song One.WAV", "Song One"),
            ("  spaced   out  .flac", "spaced out"),
            ("", "Untitled"),
            ("\x00\x1b[0m.wav", "[0m"),
            ("Bài hát Việt.m4a", "Bài hát Việt"),
        ],
    )
    def test_strips_paths_extension_and_control_characters(self, raw, expected):
        assert safe_title(raw) == expected

    def test_is_cut_to_300_characters(self):
        assert len(safe_title("x" * 1000 + ".mp3")) == 300
