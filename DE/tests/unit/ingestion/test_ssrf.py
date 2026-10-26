"""Adversarial URL suite for the SSRF guard (FR-20, NFR-16): every case must be refused."""

import pytest

from src.analysis_contract.models import ErrorCode
from src.ingestion.application.ssrf import check_url, is_public_ip
from src.ingestion.domain.errors import IngestionError

PUBLIC = "93.184.216.34"


def resolving_to(*ips: str):
    calls: list[tuple[str, int]] = []

    async def resolve(host: str, port: int) -> list[str]:
        calls.append((host, port))
        return list(ips)

    resolve.calls = calls  # type: ignore[attr-defined]
    return resolve


async def never_called(host: str, port: int) -> list[str]:
    raise AssertionError(f"resolver must not be consulted for {host}")


BLOCKED_BY_SHAPE = [
    "file:///etc/passwd",
    "ftp://example.com/x.mp3",
    "gopher://example.com/",
    "javascript:alert(1)",
    "data:audio/mp3;base64,AAAA",
    "//example.com/x.mp3",
    "example.com/x.mp3",
    "http://",
    "http://user:pass@example.com/x.mp3",
    "http://user@example.com/x.mp3",
    "http://example.com:22/x.mp3",
    "http://example.com:99999/x.mp3",
    "http://exa mple.com/x.mp3",
    "http://example.com\n/x.mp3",
    "http://localhost/",
    "http://localhost./",
    "http://LOCALHOST/",
    "http://api.localhost/",
]

BLOCKED_IP_LITERALS = [
    "http://127.0.0.1/",
    "http://127.1/",
    "http://2130706433/",
    "http://0x7f000001/",
    "http://0177.0.0.1/",
    "http://0.0.0.0/",
    "http://[::1]/",
    "http://[::]/",
    "http://[::ffff:127.0.0.1]/",
    "http://[::ffff:7f00:1]/",
    "http://10.0.0.5/",
    "http://172.16.0.1/",
    "http://172.31.255.255/",
    "http://192.168.1.1/",
    "http://169.254.169.254/latest/meta-data/",
    "http://100.64.0.1/",
    "http://224.0.0.1/",
    "http://255.255.255.255/",
    "http://[fd00::1]/",
    "http://[fe80::1]/",
    "http://[fe80::1%25eth0]/",
    "http://[64:ff9b::7f00:1]/",
    "http://[2002:7f00:1::]/",
    "http://[fd00:ec2::254]/",
]


@pytest.mark.parametrize("url", BLOCKED_BY_SHAPE)
async def test_malformed_or_non_http_urls_are_blocked_before_any_lookup(url):
    with pytest.raises(IngestionError) as exc:
        await check_url(url, never_called)
    assert exc.value.code == ErrorCode.SSRF_BLOCKED
    assert exc.value.message


@pytest.mark.parametrize("url", BLOCKED_IP_LITERALS)
async def test_non_public_ip_literals_are_blocked_without_dns(url):
    with pytest.raises(IngestionError) as exc:
        await check_url(url, never_called)
    assert exc.value.code == ErrorCode.SSRF_BLOCKED


@pytest.mark.parametrize(
    "private_ip", ["127.0.0.1", "10.1.2.3", "169.254.169.254", "::1", "fd00::1", "::ffff:10.0.0.1"]
)
async def test_hostnames_resolving_to_non_public_addresses_are_blocked(private_ip):
    with pytest.raises(IngestionError) as exc:
        await check_url("https://evil.example/x.mp3", resolving_to(private_ip))
    assert exc.value.code == ErrorCode.SSRF_BLOCKED


async def test_one_private_answer_poisons_a_mixed_answer_set():
    with pytest.raises(IngestionError) as exc:
        await check_url("https://evil.example/x.mp3", resolving_to(PUBLIC, "10.0.0.5"))
    assert exc.value.code == ErrorCode.SSRF_BLOCKED


async def test_unresolvable_hosts_are_reported_as_unresolvable_not_blocked():
    async def failing(host: str, port: int) -> list[str]:
        raise OSError("name or service not known")

    for resolve in (failing, resolving_to()):
        with pytest.raises(IngestionError) as exc:
            await check_url("https://nope.example/x.mp3", resolve)
        assert exc.value.code == ErrorCode.URL_UNRESOLVABLE


async def test_public_hostname_is_accepted_and_the_first_address_is_pinned():
    resolve = resolving_to(PUBLIC, "93.184.216.35")
    target = await check_url("https://Example.COM/a/b.mp3?x=1", resolve)
    assert (target.host, target.port, target.pinned_ip) == ("example.com", 443, PUBLIC)
    assert target.url == "https://Example.COM/a/b.mp3?x=1"
    assert resolve.calls == [("example.com", 443)]


@pytest.mark.parametrize(
    ("url", "port"), [("http://example.com/x.mp3", 80), ("https://example.com:443/x.mp3", 443)]
)
async def test_only_standard_ports_are_allowed(url, port):
    target = await check_url(url, resolving_to(PUBLIC))
    assert target.port == port


async def test_public_ip_literals_need_no_dns():
    target = await check_url(f"http://{PUBLIC}/x.mp3", never_called)
    assert (target.pinned_ip, target.port) == (PUBLIC, 80)
    v6 = await check_url("https://[2606:2800:220:1:248:1893:25c8:1946]/x.mp3", never_called)
    assert v6.pinned_ip == "2606:2800:220:1:248:1893:25c8:1946"


async def test_trailing_dot_is_normalised_away_for_public_hosts():
    target = await check_url("https://example.com./x.mp3", resolving_to(PUBLIC))
    assert target.host == "example.com"


@pytest.mark.parametrize("ip", ["8.8.8.8", "93.184.216.34", "2606:4700:4700::1111"])
def test_public_addresses_are_recognised(ip):
    assert is_public_ip(ip) is True


@pytest.mark.parametrize(
    "ip",
    ["127.0.0.1", "10.0.0.1", "192.168.0.1", "169.254.1.1", "::1", "fe80::1", "::ffff:127.0.0.1"],
)
def test_non_public_addresses_are_recognised(ip):
    assert is_public_ip(ip) is False


def test_garbage_is_not_public():
    assert is_public_ip("not-an-ip") is False
