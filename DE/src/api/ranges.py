"""HTTP ``Range`` header handling for a single stored file (RFC 9110 §14)."""

import re
from typing import Literal

_SPEC = re.compile(r"^(\d*)-(\d*)$")
UNSATISFIABLE: Literal["unsatisfiable"] = "unsatisfiable"


def parse_range(header: str | None, size: int) -> tuple[int, int] | Literal["unsatisfiable"] | None:
    """The inclusive ``(first, last)`` byte span to send, ``None`` to send the whole file, or
    ``UNSATISFIABLE`` (416).

    A header we do not serve is ignored, which RFC 9110 allows: other units, malformed specs,
    ``last < first``, and multi-range requests (answered with the full file)."""
    if not header or not header.startswith("bytes="):
        return None
    spec = header.removeprefix("bytes=").strip()
    if "," in spec:
        return None
    match = _SPEC.fullmatch(spec)
    if match is None or (match.group(1) == "" and match.group(2) == ""):
        return None
    first_text, last_text = match.groups()

    if first_text == "":  # suffix: the last N bytes
        count = int(last_text)
        if count == 0 or size == 0:
            return UNSATISFIABLE
        return max(size - count, 0), size - 1

    first = int(first_text)
    if last_text and int(last_text) < first:
        return None
    if first >= size:
        return UNSATISFIABLE
    last = size - 1 if last_text == "" else min(int(last_text), size - 1)
    return first, last
