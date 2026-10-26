"""Download a vetted URL without ever trusting DNS twice or following a redirect blindly."""

import asyncio
from pathlib import Path
from urllib.parse import urljoin

import httpx

from src.analysis_contract.models import ErrorCode
from src.ingestion.application.ssrf import Resolver, check_url, system_resolver
from src.ingestion.domain.errors import IngestionError

_REDIRECTS = {301, 302, 303, 307, 308}
_AUDIO_TYPES = {"application/octet-stream", "application/ogg"}
_CHUNK = 64 * 1024


def _unresolvable(reason: str) -> IngestionError:
    return IngestionError(ErrorCode.URL_UNRESOLVABLE, reason)


def _is_audio(content_type: str) -> bool:
    media_type = content_type.split(";")[0].strip().lower()
    return not media_type or media_type.startswith("audio/") or media_type in _AUDIO_TYPES


class SafeFetcher:
    """Every hop is validated by ``check_url`` and the connection goes to the address that
    was validated (``Host``/SNI keep the original name), so a second DNS answer is never used.
    """

    def __init__(
        self,
        resolve: Resolver = system_resolver,
        transport: httpx.AsyncBaseTransport | None = None,
        *,
        max_bytes: int,
        timeout_s: float = 30,
        max_redirects: int = 3,
    ) -> None:
        self._resolve = resolve
        self._transport = transport
        self._max_bytes = max_bytes
        self._timeout_s = timeout_s
        self._max_redirects = max_redirects

    async def fetch(self, url: str, dest: Path) -> Path:
        try:
            async with asyncio.timeout(self._timeout_s):
                return await self._fetch(url, dest)
        except TimeoutError:
            dest.unlink(missing_ok=True)
            raise _unresolvable("the server took too long to answer") from None
        except IngestionError:
            dest.unlink(missing_ok=True)
            raise
        except (httpx.HTTPError, httpx.InvalidURL):
            dest.unlink(missing_ok=True)
            raise _unresolvable("the server could not be reached") from None

    async def _fetch(self, url: str, dest: Path) -> Path:
        current = url
        async with httpx.AsyncClient(
            transport=self._transport, follow_redirects=False, trust_env=False
        ) as client:
            for _ in range(self._max_redirects + 1):
                target = await check_url(current, self._resolve)
                pinned = httpx.URL(current).copy_with(host=target.pinned_ip)
                extensions = {"sni_hostname": target.host} if pinned.scheme == "https" else {}
                async with client.stream(
                    "GET",
                    pinned,
                    headers={"Host": target.host, "Accept": "audio/*"},
                    extensions=extensions,
                ) as response:
                    if response.status_code in _REDIRECTS:
                        location = response.headers.get("location")
                        if not location:
                            raise _unresolvable("the server redirected without a destination")
                        current = urljoin(current, location)
                        continue
                    await self._save(response, dest)
                    return dest
        raise _unresolvable("too many redirects")

    async def _save(self, response: httpx.Response, dest: Path) -> None:
        if response.status_code != 200:
            raise _unresolvable(f"the server answered HTTP {response.status_code}")
        if not _is_audio(response.headers.get("content-type", "")):
            raise IngestionError(ErrorCode.UNSUPPORTED_FORMAT, "the link does not point to audio")
        declared = response.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > self._max_bytes:
            raise IngestionError(ErrorCode.FILE_TOO_LARGE, "the audio is larger than the limit")
        written = 0
        with dest.open("wb") as out:
            async for chunk in response.aiter_bytes(_CHUNK):
                written += len(chunk)
                if written > self._max_bytes:
                    raise IngestionError(
                        ErrorCode.FILE_TOO_LARGE, "the audio is larger than the limit"
                    )
                out.write(chunk)
