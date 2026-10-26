"""Removing work directories that a finished (or vanished) job left behind (task 2.8).

Every job works in ``DATA_DIR/tmp/<job id>/`` and normally cleans up after itself; this is
the safety net for crashes and killed workers. It only ever looks at UUID-named directories
directly under that one root, so ``audio/`` and ``demo/`` (siblings of ``tmp/``) are out of reach.
"""

import asyncio
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import structlog

from src.analysis_contract.models import JobState
from src.ingestion.domain.ports import JobRepository

log = structlog.get_logger("ingestion")
_UNFINISHED = (JobState.QUEUED, JobState.PROCESSING)


def _scan(root: Path) -> list[tuple[UUID, Path, float]]:
    """UUID-named real directories under ``root`` with their last-modified time."""
    if not root.is_dir():
        return []
    found = []
    for entry in root.iterdir():
        if entry.is_symlink() or not entry.is_dir():
            continue
        try:
            found.append((UUID(entry.name), entry, entry.stat().st_mtime))
        except ValueError:
            continue
    return found


def _remove(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


class TempSweeper:
    def __init__(
        self,
        root: Path,
        ttl_s: float,
        jobs: JobRepository,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._root = root
        self._ttl_s = ttl_s
        self._jobs = jobs
        self._clock = clock

    async def sweep(self) -> int:
        """Delete stale directories whose job is not running; return how many were removed."""
        now = self._clock().timestamp()
        removed = 0
        for job_id, path, modified in await asyncio.to_thread(_scan, self._root):
            if now - modified <= self._ttl_s:
                continue
            job = await self._jobs.get(job_id)
            if job is not None and job.state in _UNFINISHED:
                continue
            await asyncio.to_thread(_remove, path)
            removed += 1
        return removed

    async def run_forever(self, interval_s: float) -> None:
        while True:
            try:
                if removed := await self.sweep():
                    log.info("temp_dirs_swept", removed=removed)
            except Exception:
                log.exception("temp_sweep_failed")  # the next pass tries again
            await asyncio.sleep(interval_s)
