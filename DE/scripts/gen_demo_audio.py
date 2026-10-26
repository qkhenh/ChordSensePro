"""Render the demo tracks' audio into DATA_DIR/demo (gitignored; needs no extra packages).

    rtk uv run python scripts/gen_demo_audio.py [demo_id ...]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.ingestion.application.demo import DemoCatalog  # noqa: E402
from src.ingestion.application.demo_audio import generate_demo_audio  # noqa: E402
from src.ingestion.infrastructure.mock_engine import FIXTURES_DIR  # noqa: E402
from src.ingestion.settings import IngestSettings  # noqa: E402


def main(ids: list[str]) -> None:
    catalog = DemoCatalog(IngestSettings().data_dir / "demo", FIXTURES_DIR)
    for path in generate_demo_audio(catalog, ids or None):
        print(f"wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
