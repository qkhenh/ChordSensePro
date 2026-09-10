"""Wrapper for dag_ingest_raw — python_callables for PythonOperator.

Flow: pick_urls → download_and_stage → send_summary → trigger_dag_process

DAG 1 responsibility: Download audio → stage WAV + metadata to MongoDB.
At the end, triggers dag_process_audio via TriggerDagRunOperator (defined in DAG file).
"""
from __future__ import annotations
import asyncio
import logging
import uuid

from src.data_ingest.infrastructure.repositories.crawl_queue_repository import CrawlQueueRepository
from src.data_ingest.domain.models.crawl_queue_item import CrawlStatus
from src.data_ingest.domain.models.raw_audio_job import RawIngestOutcome
from src.data_ingest.application.ingest_raw_service import IngestRawService
from src.shared.application.email_utils import notify_dag_failure, notify_batch_summary

log = logging.getLogger(__name__)

BATCH_SIZE = 10


# ── Airflow failure callback ──────────────────────────────────────────────────

def on_task_failure(context: dict) -> None:
    """Called by Airflow when any task in dag_ingest_raw fails."""
    notify_dag_failure(
        dag_id=context["dag"].dag_id,
        task_id=context["task_instance"].task_id,
        error=str(context.get("exception", "Unknown error")),
    )


# ── Task callables ────────────────────────────────────────────────────────────

def pick_urls(**context) -> None:
    """Task 1: Fetch N pending URLs from crawl_queue, push to XCom.

    Marks each item RUNNING to prevent concurrent DAG runs from picking the same URL.
    Passes crawl_queue_id so DAG 2 can close the loop and mark DONE/FAILED.
    """
    repo  = CrawlQueueRepository()
    items = repo.get_pending(limit=BATCH_SIZE)
    for item in items:
        repo.update_status(item.id, CrawlStatus.RUNNING)

    urls = [
        {
            "crawl_queue_id": item.id,
            "source_url":     item.source_url,
            "source_type":    item.source_type,
            "annotation_url": item.annotation_url,
        }
        for item in items
    ]

    log.info(f"[pick_urls] Picked {len(urls)} URLs for this batch")
    context["ti"].xcom_push(key="urls", value=urls)


def download_and_stage(**context) -> None:
    """Task 2: Download each URL and save RawAudioJob to MongoDB.

    Uses RawIngestResult (STAGED | SKIPPED | FAILED) — no ambiguous None returns.
    crawl_queue rows are updated to FAILED/SKIPPED here; DONE update happens in DAG 2.
    """
    ti   = context["ti"]
    urls = ti.xcom_pull(key="urls") or []

    if not urls:
        log.info("[download_and_stage] No URLs to process")
        ti.xcom_push(key="results", value=[])
        return

    async def _process_all() -> list[dict]:
        service = IngestRawService()
        repo    = CrawlQueueRepository()
        results = []

        for entry in urls:
            crawl_queue_id = entry["crawl_queue_id"]
            source_url     = entry["source_url"]
            source_type    = entry.get("source_type", "local")
            analysis_id    = str(uuid.uuid4())

            log.info(f"[download_and_stage] Downloading ({source_type}): {source_url}")
            try:
                result = await service.run(
                    analysis_id=analysis_id,
                    source_url=source_url,
                    crawl_queue_id=crawl_queue_id,
                    source_type=source_type,
                    annotation_url=entry.get("annotation_url", ""),
                )

                if result.outcome == RawIngestOutcome.FAILED:
                    repo.update_status(crawl_queue_id, CrawlStatus.FAILED, result.error)
                    results.append({"url": source_url, "status": "failed", "error": result.error})
                    log.warning(f"[download_and_stage] ✗ Failed: {source_url} — {result.error}")

                elif result.outcome == RawIngestOutcome.SKIPPED:
                    repo.update_status(crawl_queue_id, CrawlStatus.SKIPPED)
                    results.append({"url": source_url, "status": "skipped"})
                    log.info(f"[download_and_stage] ~ Skipped (already staged): {source_url}")

                else:  # STAGED — leave crawl_queue as RUNNING; DAG 2 will mark DONE
                    results.append({
                        "url": source_url,
                        "status": "staged",
                        "job_id": result.job.id if result.job else "",
                    })
                    log.info(f"[download_and_stage] ✓ Staged: {source_url}")

            except Exception as e:
                error_msg = str(e)
                repo.update_status(crawl_queue_id, CrawlStatus.FAILED, error_msg)
                results.append({"url": source_url, "status": "failed", "error": error_msg})
                log.error(f"[download_and_stage] ✗ Exception: {source_url} — {error_msg}")

        return results

    results = asyncio.run(_process_all())
    ti.xcom_push(key="results", value=results)


def send_summary(**context) -> None:
    """Task 3: Log + email summary of this ingest batch."""
    ti      = context["ti"]
    results = ti.xcom_pull(key="results") or []

    staged  = [r for r in results if r["status"] == "staged"]
    failed  = [r for r in results if r["status"] == "failed"]
    skipped = [r for r in results if r["status"] == "skipped"]

    log.info(
        f"\n{'='*60}\n"
        f"Ingest raw — {len(staged)} staged, {len(failed)} failed, {len(skipped)} skipped\n"
        f"{'='*60}"
    )
    if failed:
        for f in failed:
            log.warning(f"  FAILED: {f['url']} — {f.get('error', 'unknown')}")

    notify_batch_summary("dag_ingest_raw", len(staged), len(failed), len(skipped))
