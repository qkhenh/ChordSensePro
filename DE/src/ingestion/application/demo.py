"""The demo catalog (UC1 V1): the seven golden songs, ready to try without a quota."""

from dataclasses import dataclass
from pathlib import Path

from src.analysis_contract.models import AnalysisResult

ARTIST = "ChordSense demo"
# id -> display title, in the order the catalog is listed.
_TITLES = {
    "ii_v_i_c": "ii–V–I in C",
    "blues_f": "12-bar blues in F",
    "autumn_leaves_gm": "Autumn Leaves (G minor)",
    "pop_slash_c": "Pop progression with slash chords",
    "waltz_g_3_4": "Waltz in G (3/4)",
    "compound_6_8_e": "Compound meter 6/8 in E minor",
    "rubato_downbeats_c": "Rubato with marked downbeats",
}


@dataclass(frozen=True)
class DemoTrack:
    id: str
    title: str
    artist: str
    duration_ms: int
    tempo_bpm: float
    key: str


class DemoCatalog:
    def __init__(self, audio_dir: Path, fixtures_dir: Path) -> None:
        self._audio_dir = audio_dir
        self._results = {
            demo_id: AnalysisResult.model_validate_json(
                (fixtures_dir / f"{demo_id}.result.json").read_text(encoding="utf-8")
            )
            for demo_id in _TITLES
        }

    def tracks(self) -> list[DemoTrack]:
        return [self._track(demo_id) for demo_id in _TITLES]

    def get(self, demo_id: str) -> DemoTrack | None:
        return self._track(demo_id) if demo_id in _TITLES else None

    def result(self, demo_id: str) -> AnalysisResult:
        return self._results[demo_id]

    def audio_path(self, demo_id: str) -> Path:
        """Named ``demo-<id>.wav``: the mock engine picks the golden result by that stem."""
        return self._audio_dir / f"demo-{demo_id}.wav"

    def _track(self, demo_id: str) -> DemoTrack:
        meta = self._results[demo_id].metadata
        return DemoTrack(
            id=demo_id,
            title=_TITLES[demo_id],
            artist=ARTIST,
            duration_ms=meta.duration_ms,
            tempo_bpm=meta.beat_grid.tempo_bpm,
            key=f"{meta.key.tonic.value} {meta.key.mode}",
        )
