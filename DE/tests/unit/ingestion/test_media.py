import pytest

from src.analysis_contract.models import ErrorCode, PlaybackSource
from src.ingestion.domain.errors import IngestionError
from src.ingestion.domain.naming import safe_stem
from src.ingestion.infrastructure.media import FakeMediaResolver, playback_for_url

VIDEO = "dQw4w9WgXcQ"


@pytest.mark.parametrize(
    ("url", "kind", "audio_url", "embed_id"),
    [
        (f"https://www.youtube.com/watch?v={VIDEO}&t=10", "youtube", None, VIDEO),
        (f"https://youtu.be/{VIDEO}?si=abc", "youtube", None, VIDEO),
        (f"https://music.youtube.com/watch?v={VIDEO}", "youtube", None, VIDEO),
        (f"https://www.youtube.com/shorts/{VIDEO}", "youtube", None, VIDEO),
        (f"https://www.youtube.com/embed/{VIDEO}", "youtube", None, VIDEO),
        (
            "https://soundcloud.com/artist/track?utm=x",
            "soundcloud",
            None,
            "https://soundcloud.com/artist/track",
        ),
        (
            "https://cdn.example.com/a/song.mp3",
            "direct",
            "https://cdn.example.com/a/song.mp3",
            None,
        ),
        ("https://example.org/stream", "direct", "https://example.org/stream", None),
    ],
)
def test_playback_follows_the_source(url, kind, audio_url, embed_id):
    playback = playback_for_url(url)
    assert isinstance(playback, PlaybackSource)
    assert (playback.kind, playback.audio_url, playback.embed_id) == (kind, audio_url, embed_id)


@pytest.mark.parametrize(
    "url",
    ["https://www.youtube.com/", "https://www.youtube.com/watch?v=short", "https://youtu.be/"],
)
def test_a_youtube_link_without_a_video_id_is_unresolvable(url):
    with pytest.raises(IngestionError) as exc:
        playback_for_url(url)
    assert exc.value.code == ErrorCode.URL_UNRESOLVABLE


async def test_fake_resolver_hands_back_the_demo_audio_with_a_title_from_the_url(tmp_path):
    demo = tmp_path / "demo.wav"
    demo.write_bytes(b"RIFFdemo")
    out = tmp_path / "work"
    out.mkdir()
    media = await FakeMediaResolver(demo).resolve("https://cdn.example.com/a/My%20Song.mp3", out)
    assert media.path.parent == out
    assert media.path.read_bytes() == b"RIFFdemo"
    assert media.title == "My Song"
    assert media.playback.kind == "direct"


async def test_fake_resolver_title_falls_back_to_the_host(tmp_path):
    demo = tmp_path / "demo.wav"
    demo.write_bytes(b"x")
    media = await FakeMediaResolver(demo).resolve("https://example.com/", tmp_path)
    assert media.title == "example.com"


async def test_fake_resolver_can_simulate_an_unresolvable_link(tmp_path):
    demo = tmp_path / "demo.wav"
    demo.write_bytes(b"x")
    with pytest.raises(IngestionError) as exc:
        await FakeMediaResolver(demo).resolve("https://example.com/fail-unresolvable.mp3", tmp_path)
    assert exc.value.code == ErrorCode.URL_UNRESOLVABLE


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("My Song (Live).mp3", "my-song-live"),
        ("../../etc/passwd.mp3", "passwd"),
        ("C:\\Users\\x\\fail-ai.wav", "fail-ai"),
        ("ünïcode ñame.flac", "n-code-ame"),
        ("", "audio"),
        ("....", "audio"),
        ("x" * 500 + ".mp3", "x" * 60),
        ("a\x00b\x1b[31m.wav", "a-b-31m"),
    ],
)
def test_safe_stem_is_a_short_path_safe_slug(raw, expected):
    assert safe_stem(raw) == expected


async def test_without_a_demo_file_the_resolver_synthesises_a_valid_tone_per_link(tmp_path):
    import wave

    resolver = FakeMediaResolver(None)
    first = tmp_path / "a"
    second = tmp_path / "b"
    first.mkdir()
    second.mkdir()
    one = await resolver.resolve("https://example.com/one.mp3", first)
    again = await resolver.resolve("https://example.com/one.mp3", second)
    other = await resolver.resolve("https://example.com/two.mp3", second)

    with wave.open(str(one.path)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getnframes()) == (44100, 1, 3 * 44100)
    assert one.path.read_bytes() == again.path.read_bytes()  # same link, same audio
    assert one.path.read_bytes() != other.path.read_bytes()  # different link, different audio
