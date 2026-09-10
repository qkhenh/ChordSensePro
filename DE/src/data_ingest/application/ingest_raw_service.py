"""IngestRawService — DAG 1 service: download audio and stage to MongoDB.

Responsibility: Extract only (E phase).
Does NOT run Demucs or any processing — just downloads WAV (+ annotation) and records the job.
"""
from __future__ import annotations
import os
import urllib.request
from pathlib import Path

from src.data_loader.domain.models.audio_file import AudioFile, AudioSource
from src.data_loader.application.entrypoints import run_data_loader
from src.data_ingest.domain.models.raw_audio_job import RawAudioJob, RawIngestResult
from src.data_ingest.infrastructure.mongo.raw_audio_job_repository import RawAudioJobRepository
from src.shared.infrastructure.mongo.client import get_mongo_db


class IngestRawService:
    """Downloads one audio URL (+ annotation file) and saves a RawAudioJob to MongoDB.

    Called by DAG 1 (dag_ingest_raw).
    Does NOT process audio — that is DAG 2's job.
    """

    def _tmp_dir(self, analysis_id: str) -> Path:
        base = Path(os.getenv("DATA_DIR", "/data")) / "tmp" / analysis_id
        base.mkdir(parents=True, exist_ok=True)
        return base

    async def run(
        self,
        analysis_id: str,
        source_url: str,
        crawl_queue_id: str,
        source_type: str = "local",
        annotation_url: str = "",
    ) -> RawIngestResult:
        """Download audio (and annotation if available), save RawAudioJob to MongoDB.

        Args:
            analysis_id:     UUID for this job (becomes MongoDB _id)
            source_url:      remote URL or local path to audio
            crawl_queue_id:  ID of the crawl_queue row — stored so DAG 2 can update it
            source_type:     AudioSource value for Downloader routing
            annotation_url:  remote URL to annotation file (.jams/.json); "" if not available

        Returns:
            RawIngestResult with outcome STAGED | SKIPPED | FAILED
        """
        repo = RawAudioJobRepository(get_mongo_db())

        # Dedup — skip if already downloaded (any status)
        if await repo.exists(source_url):
            return RawIngestResult.skipped()

        dest_dir = self._tmp_dir(analysis_id)

        # ── E: Download audio ─────────────────────────────────────────────────
        audio_file = AudioFile.from_url(
            source_url=source_url,
            analysis_id=analysis_id,
            dest_dir=dest_dir,
            source_type=AudioSource(source_type),
        )
        response = run_data_loader(audio_file)
        if response.is_err:
            return RawIngestResult.failed(response.error or "Download failed")

        # ── E: Download annotation (optional) ────────────────────────────────
        annotation_path = ""
        if annotation_url:
            annotation_path = self._download_annotation(annotation_url, dest_dir)

        # ── Stage to MongoDB ──────────────────────────────────────────────────
        job = RawAudioJob(
            id=analysis_id,
            source_url=source_url,
            source_type=source_type,
            wav_path=str(response.wav_path),
            annotation_path=annotation_path,
            crawl_queue_id=crawl_queue_id,
        )
        await repo.save(job)
        return RawIngestResult.staged(job)

    @staticmethod
    def _download_annotation(url: str, dest_dir: Path) -> str:
        """Download or copy annotation file to dest_dir.

        Handles:
        - HTTP/HTTPS URL  → urllib download (ChoCo GitHub, JAAH Zenodo)
        - Absolute path   → copy to dest_dir  (local datasets via seed script)

        Returns local path string, or '' on failure (non-fatal).
        """
        if not url:
            return ""

        # Local file path (starts with / or file://)
        src_path = url.removeprefix("file://")
        if src_path.startswith("/"):
            src = Path(src_path)
            if not src.exists():
                return ""
            import shutil
            dest = dest_dir / src.name
            shutil.copy2(src, dest)
            return str(dest)

        # HTTP/HTTPS download
        ext  = url.rsplit(".", 1)[-1] if "." in url else "json"
        dest = dest_dir / f"annotation.{ext}"
        try:
            urllib.request.urlretrieve(url, dest)
            return str(dest)
        except Exception:
            return ""  # non-fatal — AnnotationHandler skips gracefully
