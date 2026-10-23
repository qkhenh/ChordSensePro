"""Tests for the deterministic train/val/test split."""
from collections import Counter

from src.shared.domain.dataset_split import TEST, TRAIN, VAL, split_for


def test_split_is_deterministic():
    url = "https://www.youtube.com/watch?v=abc123"
    assert {split_for(url) for _ in range(10)} == {split_for(url)}


def test_split_values_are_valid():
    assert {split_for(f"song-{i}") for i in range(500)} <= {TRAIN, VAL, TEST}


def test_split_ratio_is_roughly_70_20_10():
    counts = Counter(split_for(f"https://example.com/track/{i}") for i in range(10_000))
    assert 0.66 < counts[TRAIN] / 10_000 < 0.74
    assert 0.17 < counts[VAL] / 10_000 < 0.23
    assert 0.07 < counts[TEST] / 10_000 < 0.13
