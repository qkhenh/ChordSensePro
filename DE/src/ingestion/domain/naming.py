"""Turning user-controlled names into safe ones (file stems, display titles)."""

import re
import unicodedata

MAX_TITLE_LENGTH = 300
MAX_STEM_LENGTH = 60
_UNSAFE = re.compile(r"[^a-z0-9_-]+")


def _base_name(raw: str) -> str:
    """Last path component without its extension, whichever separator style was used."""
    name = re.split(r"[\\/]", raw)[-1]
    return name.rsplit(".", 1)[0] if "." in name else name


def safe_stem(raw: str) -> str:
    """A short ``[a-z0-9_-]`` slug usable as a file name on any platform."""
    slug = _UNSAFE.sub("-", _base_name(raw).lower()).strip("-")
    return slug[:MAX_STEM_LENGTH].strip("-") or "audio"


def safe_title(raw: str) -> str:
    """A display title: no directories, extension or control characters, at most 300 chars."""
    cleaned = "".join(
        " " if unicodedata.category(c).startswith("C") else c for c in _base_name(raw)
    )
    return " ".join(cleaned.split())[:MAX_TITLE_LENGTH].strip() or "Untitled"
