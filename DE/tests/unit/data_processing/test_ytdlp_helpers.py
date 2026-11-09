"""Tests for shared yt-dlp helpers (pure functions — no network, no yt_dlp import)."""
from pathlib import Path

from src.data_loader.application.downloaders._ytdlp import base_opts, pick_best_candidate


def _e(i: str, duration):
    return {"id": i, "duration": duration}


def _pick_id(entries, expected, tolerance):
    best = pick_best_candidate(entries, expected, tolerance=tolerance)
    assert best is not None
    return best["id"]


def test_picks_closest_duration():
    entries = [_e("a", 300), _e("b", 143), _e("c", 150)]
    assert _pick_id(entries, 142.5, 3.0) == "b"


def test_none_when_outside_tolerance():
    assert pick_best_candidate([_e("a", 300), _e("b", 90)], 142.5, tolerance=3.0) is None


def test_ignores_entries_without_duration():
    entries = [_e("live", None), _e("ok", 142)]
    assert _pick_id(entries, 142.5, 3.0) == "ok"


def test_no_expected_duration_returns_first_valid():
    entries = [None, {"id": None}, _e("first", 10), _e("second", 20)]
    assert _pick_id(entries, None, 3.0) == "first"


def test_empty_results():
    assert pick_best_candidate([], 100.0, tolerance=3.0) is None
    assert pick_best_candidate([None], None, tolerance=3.0) is None


def test_base_opts_has_resilience_settings(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("YTDLP_COOKIES", raising=False)
    opts = base_opts(tmp_path)
    assert opts["retries"] >= 3 and opts["socket_timeout"] > 0
    assert "cookiefile" not in opts


def test_base_opts_uses_cookies_only_if_file_exists(tmp_path: Path, monkeypatch):
    cookies = tmp_path / "cookies.txt"
    monkeypatch.setenv("YTDLP_COOKIES", str(cookies))
    assert "cookiefile" not in base_opts(tmp_path)       # file missing → ignored
    cookies.write_text("# Netscape HTTP Cookie File")
    assert base_opts(tmp_path)["cookiefile"] == str(cookies)
