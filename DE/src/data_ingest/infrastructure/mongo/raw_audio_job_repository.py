"""MongoDB repository for raw_audio_jobs collection.

Collection: raw_audio_jobs
Purpose: Staging layer between DAG 1 (download) and DAG 2 (processing).
"""
from __future__ import annotations
from datetime import datetime, timezone

from motor.motor_asyncio import AsyncIOMotorDatabase

from src.data_ingest.domain.models.raw_audio_job import RawAudioJob, RawJobStatus


class RawAudioJobRepository:
    """Async MongoDB repository for raw_audio_jobs collection."""

    COLLECTION = "raw_audio_jobs"

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._col = db[self.COLLECTION]

    # ── Write ────────────────────────────────────────────────────────────────

    async def save(self, job: RawAudioJob) -> None:
        """Upsert a RawAudioJob document."""
        doc = {
            "_id":             job.id,
            "source_url":      job.source_url,
            "source_type":     job.source_type,
            "wav_path":        job.wav_path,
            "annotation_path": job.annotation_path,
            "crawl_queue_id":  job.crawl_queue_id,
            "status":          job.status.value,
            "error_message":   job.error_message,
            "created_at":      job.created_at,
            "updated_at":      job.updated_at,
        }
        await self._col.replace_one({"_id": job.id}, doc, upsert=True)

    async def update_status(
        self,
        job_id: str,
        status: RawJobStatus,
        error_message: str | None = None,
    ) -> None:
        """Update status (and optional error) for one job."""
        update: dict = {
            "$set": {
                "status":     status.value,
                "updated_at": datetime.now(timezone.utc),
            }
        }
        if error_message is not None:
            update["$set"]["error_message"] = error_message
        await self._col.update_one({"_id": job_id}, update)

    # ── Read ─────────────────────────────────────────────────────────────────

    async def get_pending(self, limit: int = 10) -> list[RawAudioJob]:
        """Fetch jobs with status=pending_processing, oldest first."""
        cursor = (
            self._col
            .find({"status": RawJobStatus.PENDING.value})
            .sort("created_at", 1)
            .limit(limit)
        )
        return [self._to_domain(doc) async for doc in cursor]

    async def exists(self, source_url: str) -> bool:
        """Return True if a job with this source_url already exists (any status)."""
        doc = await self._col.find_one({"source_url": source_url}, {"_id": 1})
        return doc is not None

    async def count_by_status(self) -> dict[str, int]:
        """Return {status: count} for all jobs — used by summary tasks."""
        pipeline = [{"$group": {"_id": "$status", "n": {"$sum": 1}}}]
        cursor = self._col.aggregate(pipeline)
        return {doc["_id"]: doc["n"] async for doc in cursor}

    # ── Private ──────────────────────────────────────────────────────────────

    @staticmethod
    def _to_domain(doc: dict) -> RawAudioJob:
        return RawAudioJob(
            id=doc["_id"],
            source_url=doc["source_url"],
            source_type=doc["source_type"],
            wav_path=doc["wav_path"],
            annotation_path=doc.get("annotation_path", ""),
            crawl_queue_id=doc.get("crawl_queue_id", ""),
            status=RawJobStatus(doc["status"]),
            error_message=doc.get("error_message"),
            created_at=doc.get("created_at", datetime.now(timezone.utc)),
            updated_at=doc.get("updated_at", datetime.now(timezone.utc)),
        )
