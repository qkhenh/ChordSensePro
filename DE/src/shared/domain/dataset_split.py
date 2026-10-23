"""Deterministic train / val / test split, decided once per song at ingest time.

Splitting by a hash of the song's source key (not by segment, not randomly) guarantees:
  * all segments of one song land in the same split  → no leakage between overlapping
    2-second segments of the same recording;
  * re-crawling the same source later gives the same split;
  * augmentation can be applied to the train split only, afterwards.

Ratio follows 18_DE_AI_chordsense.md: 70% train / 20% val / 10% test.
"""
from __future__ import annotations
import hashlib

TRAIN = "train"
VAL   = "val"
TEST  = "test"

_TRAIN_PCT = 70
_VAL_PCT   = 20   # remaining 10% → test


def split_for(key: str) -> str:
    """Return 'train' | 'val' | 'test' for a stable song key (use the source URL).

    Hooktheory clips of one YouTube video share ``…watch?v=<id>#ht=<clip>``: the fragment is
    dropped so every clip of the same recording lands in the same split.
    """
    if "#ht=" in key:
        key = key.split("#ht=", 1)[0]
    bucket = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < _TRAIN_PCT:
        return TRAIN
    if bucket < _TRAIN_PCT + _VAL_PCT:
        return VAL
    return TEST
