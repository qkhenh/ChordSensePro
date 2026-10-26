"""The effective-events golden file is shared with the TypeScript twin of ``apply_overrides``."""

import json
from pathlib import Path

from src.analysis_contract.models import Chart
from src.analysis_contract.overrides import apply_overrides

CHARTS = Path(__file__).parents[4] / "contracts" / "fixtures" / "charts"
EDITED = CHARTS / "autumn_leaves_gm.edited.chart.json"
EFFECTIVE = CHARTS / "autumn_leaves_gm.effective.json"


def effective_events():
    chart = Chart.model_validate_json(EDITED.read_text(encoding="utf-8"))
    return chart, apply_overrides(chart.ai_events, chart.overrides)


def test_the_golden_file_is_what_apply_overrides_produces():
    _, events = effective_events()
    expected = [e.model_dump(mode="json") for e in events]
    assert json.loads(EFFECTIVE.read_text(encoding="utf-8")) == expected


def test_the_golden_file_exercises_a_set_and_a_delete():
    chart, events = effective_events()
    baseline = {e.event_id: e for e in chart.ai_events}
    kept = {e.event_id: e for e in events}
    assert len(events) == len(chart.ai_events) - 1  # e0008 was deleted
    assert "e0008" not in kept and "e0008" in baseline
    assert kept["e0003"].symbol != baseline["e0003"].symbol and kept["e0003"].is_user_edited
