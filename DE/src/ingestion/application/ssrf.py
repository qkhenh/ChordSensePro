"""SSRF guard for user-supplied media URLs (FR-20, NFR-16).

``check_url`` accepts only plain http(s) URLs on the standard ports whose host resolves
exclusively to public addresses, and returns the address to *pin*: the fetcher must connect
to that address, never resolve again (DNS rebinding).
"""

import asyncio
import ipaddress
import re
import socket
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit

from src.analysis_contract.models import ErrorCode
from src.ingestion.domain.errors import IngestionError

Resolver = Callable[[str, int], Awaitable[list[str]]]

ALLOWED_PORTS = {"http": 80, "https": 443}
_LEGACY_IPV4 = re.compile(r"[0-9a-fx.]+")
_NAT64 = ipaddress.IPv6Network("64:ff9b::/96")


@dataclass(frozen=True)
class SafeTarget:
    url: str
    host: str
    port: int
    pinned_ip: str


def _blocked(reason: str) -> IngestionError:
    return IngestionError(ErrorCode.SSRF_BLOCKED, reason)


def is_public_ip(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    if isinstance(addr, ipaddress.IPv6Address):
        if addr.ipv4_mapped is not None:
            return is_public_ip(str(addr.ipv4_mapped))
        if addr.teredo is not None:
            return False
        if addr.sixtofour is not None:
            return is_public_ip(str(addr.sixtofour)) and addr.is_global
        if addr in _NAT64 or int(addr) < 2**32:  # NAT64 and IPv4-compatible: judge the embedded v4
            return is_public_ip(str(ipaddress.IPv4Address(int(addr) & 0xFFFFFFFF)))
    return addr.is_global and not (
        addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
        or addr.is_loopback
        or addr.is_link_local
    )


def _ip_literal(host: str) -> str | None:
    """The address a host spells out (also decimal/hex/octal/short IPv4), else None."""
    if ":" in host:
        try:
            return str(ipaddress.ip_address(unquote(host)))
        except ValueError:
            raise _blocked("the host is not a valid IPv6 address") from None
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        pass
    if _LEGACY_IPV4.fullmatch(host):
        try:
            return socket.inet_ntoa(socket.inet_aton(host))
        except OSError:
            return None
    return None


async def system_resolver(host: str, port: int) -> list[str]:
    infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return list(dict.fromkeys(str(info[4][0]) for info in infos))


async def check_url(url: str, resolve: Resolver = system_resolver) -> SafeTarget:
    if any(c.isspace() or ord(c) < 32 for c in url):
        raise _blocked("the URL contains whitespace or control characters")
    try:
        parts = urlsplit(url)
        explicit_port = parts.port
    except ValueError:
        raise _blocked("the URL is malformed") from None
    scheme = parts.scheme.lower()
    if scheme not in ALLOWED_PORTS:
        raise _blocked("only http and https links are accepted")
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise _blocked("links with embedded credentials are not accepted")
    host = (parts.hostname or "").rstrip(".")
    if not host:
        raise _blocked("the URL has no host")
    port = explicit_port or ALLOWED_PORTS[scheme]
    if port != ALLOWED_PORTS[scheme]:
        raise _blocked("only the standard ports 80 and 443 are accepted")
    if host == "localhost" or host.endswith(".localhost"):
        raise _blocked("localhost is not reachable")

    literal = _ip_literal(host)
    if literal is not None:
        if not is_public_ip(literal):
            raise _blocked("the address is private, loopback or link-local")
        return SafeTarget(url, literal, port, literal)

    try:
        ascii_host = host.encode("idna").decode()
    except UnicodeError:
        raise _blocked("the host name is not valid") from None
    try:
        addresses = await resolve(ascii_host, port)
    except OSError:
        raise IngestionError(
            ErrorCode.URL_UNRESOLVABLE, "the host name cannot be resolved"
        ) from None
    if not addresses:
        raise IngestionError(ErrorCode.URL_UNRESOLVABLE, "the host name cannot be resolved")
    if not all(is_public_ip(address) for address in addresses):
        raise _blocked("the host resolves to a private, loopback or link-local address")
    return SafeTarget(url, ascii_host, port, addresses[0])
