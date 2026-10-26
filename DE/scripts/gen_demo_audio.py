"""Render the demo tracks' audio into DATA_DIR/demo (gitignored; needs no extra packages).

    rtk uv run python scripts/gen_demo_audio.py [--force] [demo_id ...]

Files that already exist are kept unless --force is given.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.ingestion.application.demo import DemoCatalog  # noqa: E402
from src.ingestion.application.demo_audio import generate_demo_audio  # noqa: E402
from src.ingestion.infrastructure.mock_engine import FIXTURES_DIR  # noqa: E402
from src.ingestion.settings import IngestSettings  # noqa: E402


def main(args: list[str]) -> None:
    force = "--force" in args
    ids = [a for a in args if a != "--force"]
    catalog = DemoCatalog(IngestSettings().data_dir / "demo", FIXTURES_DIR)
    written = generate_demo_audio(catalog, ids or None, overwrite=force)
    for path in written:
        print(f"wrote {path}")
    if not written:
        print("nothing to do: demo audio already exists (use --force to render again)")


if __name__ == "__main__":
    main(sys.argv[1:])
