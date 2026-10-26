"""SafeFetcher: pinned connections, redirect re-validation, size/time limits."""

import asyncio

import httpx
import pytest

from src.analysis_contract.models import ErrorCode
from src.ingestion.application.fetcher import SafeFetcher
from src.ingestion.domain.errors import IngestionError

PUBLIC = "93.184.216.34"
OTHER_PUBLIC = "93.184.216.99"
URL = "https://example.com/a/song.mp3"


class Resolver:
    """Hostname -> answers; a list of answer sets is consumed one per call (rebinding)."""

    def __init__(self, answers: dict[str, list[str] | list[list[str]]]):
        self.answers = answers
        self.calls: list[str] = []

    async def __call__(self, host: str, port: int) -> list[str]:
        self.calls.append(host)
        value = self.answers[host]
        if value and isinstance(value[0], list):
            return value.pop(0) if len(value) > 1 else value[0]
        return value  # type: ignore[return-value]


def audio(body: bytes = b"abc", **headers):
    return httpx.Response(200, content=body, headers={"content-type": "audio/mpeg", **headers})


def fetcher(handler, resolver=None, **kwargs) -> tuple[SafeFetcher, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    async def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        result = handler(request)
        return await result if asyncio.iscoroutine(result) else result

    resolver = resolver or Resolver({"example.com": [PUBLIC]})
    return (
        SafeFetcher(resolver, httpx.MockTransport(record), **{"max_bytes": 1024, **kwargs}),
        seen,
    )


async def test_connects_to_the_pinned_address_and_keeps_the_original_host(tmp_path):
    f, seen = fetcher(lambda r: audio(b"hello"))
    dest = await f.fetch(URL, tmp_path / "out.bin")
    assert dest.read_bytes() == b"hello"
    (request,) = seen
    assert request.url.host == PUBLIC
    assert request.headers["host"] == "example.com"
    assert request.extensions["sni_hostname"] == "example.com"  # TLS is verified against the name
    assert request.url.path == "/a/song.mp3"


async def test_ipv6_answers_are_pinned_too(tmp_path):
    resolver = Resolver({"example.com": ["2606:2800:220:1:248:1893:25c8:1946"]})
    f, seen = fetcher(lambda r: audio(), resolver)
    await f.fetch(URL, tmp_path / "out.bin")
    assert seen[0].url.host == "2606:2800:220:1:248:1893:25c8:1946"


async def test_each_hop_resolves_exactly_once(tmp_path):
    resolver = Resolver({"example.com": [PUBLIC]})
    f, _ = fetcher(lambda r: audio(), resolver)
    await f.fetch(URL, tmp_path / "out.bin")
    assert resolver.calls == ["example.com"]


async def test_follows_relative_and_cross_host_redirects_revalidating_each(tmp_path):
    resolver = Resolver({"example.com": [PUBLIC], "cdn.example.net": [OTHER_PUBLIC]})

    def handler(request):
        if request.url.path == "/a/song.mp3":
            return httpx.Response(302, headers={"location": "/b/moved.mp3"})
        if request.url.path == "/b/moved.mp3":
            return httpx.Response(301, headers={"location": "https://cdn.example.net/final.mp3"})
        return audio(b"final")

    f, seen = fetcher(handler, resolver)
    assert (await f.fetch(URL, tmp_path / "out.bin")).read_bytes() == b"final"
    assert [r.url.host for r in seen] == [PUBLIC, PUBLIC, OTHER_PUBLIC]
    assert resolver.calls == ["example.com", "example.com", "cdn.example.net"]


@pytest.mark.parametrize(
    "location",
    [
        "http://169.254.169.254/latest/meta-data/",
        "http://127.0.0.1/admin",
        "http://localhost/x",
        "file:///etc/passwd",
        "http://[::1]/x",
        "http://2130706433/x",
    ],
)
async def test_a_redirect_into_a_private_target_is_blocked(tmp_path, location):
    f, seen = fetcher(lambda r: httpx.Response(302, headers={"location": location}))
    with pytest.raises(IngestionError) as exc:
        await f.fetch(URL, tmp_path / "out.bin")
    assert exc.value.code == ErrorCode.SSRF_BLOCKED
    assert len(seen) == 1  # the forbidden target was never contacted
    assert not (tmp_path / "out.bin").exists()


async def test_dns_rebinding_between_hops_is_caught(tmp_path):
    resolver = Resolver({"example.com": [[PUBLIC], ["10.0.0.5"]]})
    f, seen = fetcher(lambda r: httpx.Response(302, headers={"location": "/second"}), resolver)
    with pytest.raises(IngestionError) as exc:
        await f.fetch(URL, tmp_path / "out.bin")
    assert exc.value.code == ErrorCode.SSRF_BLOCKED
    assert len(seen) == 1
    assert resolver.calls == ["example.com", "example.com"]


async def test_too_many_redirects_are_refused(tmp_path):
    f, seen = fetcher(lambda r: httpx.Response(302, headers={"location": "/again"}))
    with pytest.raises(IngestionError) as exc:
        await f.fetch(URL, tmp_path / "out.bin")
    assert exc.value.code == ErrorCode.URL_UNRESOLVABLE
    assert len(seen) == 4  # the first request plus three redirects


async def test_a_redirect_without_location_is_refused(tmp_path):
    f, _ = fetcher(lambda r: httpx.Response(302))
    with pytest.raises(IngestionError) as exc:
        await f.fetch(URL, tmp_path / "out.bin")
    assert exc.value.code == ErrorCode.URL_UNRESOLVABLE


async def test_the_size_cap_is_enforced_while_streaming_and_the_partial_file_removed(tmp_path):
    f, _ = fetcher(lambda r: audio(b"x" * 2000), max_bytes=1024)
    with pytest.raises(IngestionError) as exc:
        await f.fetch(URL, tmp_path / "out.bin")
    assert exc.value.code == ErrorCode.FILE_TOO_LARGE
    assert not (tmp_path / "out.bin").exists()


async def test_a_declared_oversize_body_is_refused_before_reading_it(tmp_path):
    f, _ = fetcher(lambda r: audio(b"x", **{"content-length": "999999"}), max_bytes=1024)
    with pytest.raises(IngestionError) as exc:
        await f.fetch(URL, tmp_path / "out.bin")
    assert exc.value.code == ErrorCode.FILE_TOO_LARGE


async def test_a_body_of_exactly_the_cap_is_accepted(tmp_path):
    f, _ = fetcher(lambda r: audio(b"x" * 1024), max_bytes=1024)
    assert len((await f.fetch(URL, tmp_path / "out.bin")).read_bytes()) == 1024


async def test_a_slow_server_times_out_as_unresolvable(tmp_path):
    async def slow(request):
        await asyncio.sleep(1)
        return audio()

    f, _ = fetcher(slow, timeout_s=0.05)
    with pytest.raises(IngestionError) as exc:
        await f.fetch(URL, tmp_path / "out.bin")
    assert exc.value.code == ErrorCode.URL_UNRESOLVABLE


@pytest.mark.parametrize("status", [401, 403, 404, 410, 500, 503])
async def test_error_statuses_are_unresolvable(tmp_path, status):
    f, _ = fetcher(lambda r: httpx.Response(status))
    with pytest.raises(IngestionError) as exc:
        await f.fetch(URL, tmp_path / "out.bin")
    assert exc.value.code == ErrorCode.URL_UNRESOLVABLE


async def test_connection_failures_are_unresolvable(tmp_path):
    def refuse(request):
        raise httpx.ConnectError("refused")

    f, _ = fetcher(refuse)
    with pytest.raises(IngestionError) as exc:
        await f.fetch(URL, tmp_path / "out.bin")
    assert exc.value.code == ErrorCode.URL_UNRESOLVABLE


@pytest.mark.parametrize(
    "content_type", ["text/html; charset=utf-8", "application/json", "image/png"]
)
async def test_non_audio_content_types_are_unsupported(tmp_path, content_type):
    f, _ = fetcher(
        lambda r: httpx.Response(200, content=b"<html>", headers={"content-type": content_type})
    )
    with pytest.raises(IngestionError) as exc:
        await f.fetch(URL, tmp_path / "out.bin")
    assert exc.value.code == ErrorCode.UNSUPPORTED_FORMAT


@pytest.mark.parametrize(
    "content_type", ["audio/mpeg", "audio/mp4", "application/octet-stream", "application/ogg"]
)
async def test_audio_like_content_types_are_accepted(tmp_path, content_type):
    f, _ = fetcher(
        lambda r: httpx.Response(200, content=b"abc", headers={"content-type": content_type})
    )
    assert (await f.fetch(URL, tmp_path / "out.bin")).exists()


async def test_the_blocked_url_itself_never_reaches_the_network(tmp_path):
    f, seen = fetcher(lambda r: audio())
    with pytest.raises(IngestionError) as exc:
        await f.fetch("http://169.254.169.254/latest/meta-data/", tmp_path / "out.bin")
    assert exc.value.code == ErrorCode.SSRF_BLOCKED
    assert seen == []
