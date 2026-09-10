"""Wrapper for dag_process_audio — python_callables for PythonOperator.

Flow: pick_pending_jobs → process_and_save → send_summary

DAG 2 responsibility: Read RawAudioJob from MongoDB → Demucs → features → PostgreSQL.
After each job, updates BOTH MongoDB (job status) AND Postgres crawl_queue (via crawl_queue_id).
Triggered by dag_ingest_raw via TriggerDagRunOperator — not on a time schedule.
"""
from __future__ import annotations
import asyncio
import logging

from src.data_ingest.infrastructure.mongo.raw_audio_job_repository import RawAudioJobRepository
from src.data_ingest.domain.models.raw_audio_job import RawAudioJob, RawJobStatus
from src.data_ingest.application.process_audio_service import ProcessAudioService
from src.data_ingest.domain.models.ingest_record import IngestStatus
from src.data_ingest.infrastructure.repositories.crawl_queue_repository import CrawlQueueRepository
from src.data_ingest.domain.models.crawl_queue_item import CrawlStatus
from src.shared.infrastructure.mongo.client import get_mongo_db
from src.shared.application.email_utils import notify_dag_failure, notify_batch_summary

log = logging.getLogger(__name__)

BATCH_SIZE = 10


# ── Airflow failure callback ──────────────────────────────────────────────────

def on_task_failure(context: dict) -> None:
    """Called by Airflow when any task in dag_process_audio fails."""
    notify_dag_failure(
        dag_id=context["dag"].dag_id,
        task_id=context["task_instance"].task_id,
        error=str(context.get("exception", "Unknown error")),
    )


# ── Task callables ────────────────────────────────────────────────────────────

def pick_pending_jobs(**context) -> None:
    """Task 1: Fetch N pending RawAudioJobs from MongoDB, push to XCom.

    Marks each job PROCESSING to prevent duplicate pickup.
    Includes crawl_queue_id so process_and_save can close the Postgres loop.
    """
    async def _run():
        repo = RawAudioJobRepository(get_mongo_db())
        jobs = await repo.get_pending(limit=BATCH_SIZE)
        for job in jobs:
            await repo.update_status(job.id, RawJobStatus.PROCESSING)
        return [
            {
                "id":              job.id,
                "source_url":      job.source_url,
                "source_type":     job.source_type,
                "wav_path":        job.wav_path,
                "crawl_queue_id":  job.crawl_queue_id,
            }
            for job in jobs
        ]

    jobs = asyncio.run(_run())
    log.info(f"[pick_pending_jobs] Picked {len(jobs)} jobs for processing")
    context["ti"].xcom_push(key="jobs", value=jobs)


def process_and_save(**context) -> None:
    """Task 2: Run Demucs + features + save to PostgreSQL for each pending job.

    On completion: updates MongoDB job status AND crawl_queue (via crawl_queue_id).
    """
    ti   = context["ti"]
    jobs = ti.xcom_pull(key="jobs") or []

    if not jobs:
        log.info("[process_and_save] No jobs to process")
        ti.xcom_push(key="results", value=[])
        return

    async def _process_all() -> list[dict]:
        service    = ProcessAudioService()
        mongo_repo = RawAudioJobRepository(get_mongo_db())
        cq_repo    = CrawlQueueRepository()
        results    = []

        for entry in jobs:
            job = RawAudioJob(
                id=entry["id"],
                source_url=entry["source_url"],
                source_type=entry["source_type"],
                wav_path=entry["wav_path"],
                crawl_queue_id=entry.get("crawl_queue_id", ""),
                status=RawJobStatus.PROCESSING,
            )
            log.info(f"[process_and_save] Processing: {job.source_url}")

            try:
                record = await service.run(job)

                # ── Update Postgres crawl_queue — closes the RUNNING loop ────
                if job.crawl_queue_id:
                    if record.status in (IngestStatus.DONE,):
                        cq_repo.update_status(job.crawl_queue_id, CrawlStatus.DONE)
                    elif record.status == IngestStatus.FAILED:
                        cq_repo.update_status(
                            job.crawl_queue_id,
                            CrawlStatus.FAILED,
                            record.error_message,
                        )

                if record.status == IngestStatus.DONE:
                    results.append({"url": job.source_url, "status": "done"})
                    log.info(f"[process_and_save] ✓ Done: {job.source_url}")

                elif record.status == IngestStatus.FAILED:
                    results.append({
                        "url": job.source_url,
                        "status": "failed",
                        "error": record.error_message,
                    })
                    log.warning(f"[process_and_save] ✗ Failed: {job.source_url} — {record.error_message}")

                else:
                    results.append({"url": job.source_url, "status": "skipped"})
                    log.info(f"[process_and_save] ~ Skipped: {job.source_url}")

            except Exception as e:
                error_msg = str(e)
                await mongo_repo.update_status(job.id, RawJobStatus.FAILED, error_msg)
                if job.crawl_queue_id:
                    cq_repo.update_status(job.crawl_queue_id, CrawlStatus.FAILED, error_msg)
                results.append({"url": job.source_url, "status": "failed", "error": error_msg})
                log.error(f"[process_and_save] ✗ Exception: {job.source_url} — {error_msg}")

        return results

    results = asyncio.run(_process_all())
    ti.xcom_push(key="results", value=results)


def send_summary(**context) -> None:
    """Task 3: Log + email summary of this processing batch."""
    ti      = context["ti"]
    results = ti.xcom_pull(key="results") or []

    done    = [r for r in results if r["status"] == "done"]
    failed  = [r for r in results if r["status"] == "failed"]
    skipped = [r for r in results if r["status"] == "skipped"]

    log.info(
        f"\n{'='*60}\n"
        f"Process audio — {len(done)} done, {len(failed)} failed, {len(skipped)} skipped\n"
        f"{'='*60}"
    )
    if failed:
        for f in failed:
            log.warning(f"  FAILED: {f['url']} — {f.get('error', 'unknown')}")

    notify_batch_summary("dag_process_audio", len(done), len(failed), len(skipped))
