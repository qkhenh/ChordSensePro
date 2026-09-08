"""Wrapper for dag_crawl_batch — python_callables for PythonOperator.

Flow:
    pick_urls  → run_etl  → send_summary
"""
from __future__ import annotations
import asyncio
import logging
import uuid

from src.shared.infrastructure.postgres.client import get_session
from src.data_ingest.infrastructure.repositories.crawl_queue_repository import CrawlQueueRepository
from src.data_ingest.domain.models.crawl_queue_item import CrawlStatus
from src.data_ingest.application.entrypoints import run_ingest_pipeline
from src.data_ingest.domain.models.ingest_record import IngestStatus

log = logging.getLogger(__name__)

BATCH_SIZE = 10   # URLs per DAG run — tune based on processing time


def pick_urls(**context) -> None:
    """Task 1: Fetch N pending URLs from crawl_queue, push to XCom.

    Marks selected items as RUNNING so concurrent DAG runs don't pick same URLs.
    """
    async def _run():
        async with get_session() as session:
            repo = CrawlQueueRepository(session)
            items = await repo.get_pending(limit=BATCH_SIZE)

            # Mark as RUNNING immediately to prevent double-pickup
            for item in items:
                await repo.update_status(item.id, CrawlStatus.RUNNING)

            return [{"id": item.id, "source_url": item.source_url} for item in items]

    urls = asyncio.run(_run())
    log.info(f"[pick_urls] Picked {len(urls)} URLs for this batch run")
    context["ti"].xcom_push(key="urls", value=urls)


def run_etl(**context) -> None:
    """Task 2: Run E→T→L for each URL. Update crawl_queue status per result.

    Processes sequentially. Each URL is independent — one failure does not stop others.
    Pushes per-URL results to XCom for summary task.
    """
    ti   = context["ti"]
    urls = ti.xcom_pull(key="urls") or []

    if not urls:
        log.info("[run_etl] No URLs to process — skipping")
        ti.xcom_push(key="results", value=[])
        return

    results = []

    async def _update(item_id: str, status: CrawlStatus, error: str | None = None):
        async with get_session() as session:
            repo = CrawlQueueRepository(session)
            await repo.update_status(item_id, status, error)

    for entry in urls:
        item_id    = entry["id"]
        source_url = entry["source_url"]
        analysis_id = str(uuid.uuid4())

        log.info(f"[run_etl] Processing: {source_url}")
        try:
            record = run_ingest_pipeline(
                analysis_id=analysis_id,
                source_url=source_url,
                user_id="",   # ETL batch — no user
            )

            if record.status == IngestStatus.DONE:
                asyncio.run(_update(item_id, CrawlStatus.DONE))
                results.append({"url": source_url, "status": "done"})
                log.info(f"[run_etl] ✓ Done: {source_url}")

            elif record.status == IngestStatus.FAILED:
                asyncio.run(_update(item_id, CrawlStatus.FAILED, record.error_message))
                results.append({"url": source_url, "status": "failed", "error": record.error_message})
                log.warning(f"[run_etl] ✗ Failed: {source_url} — {record.error_message}")

            else:
                # Skipped (already analyzed)
                asyncio.run(_update(item_id, CrawlStatus.SKIPPED))
                results.append({"url": source_url, "status": "skipped"})
                log.info(f"[run_etl] ~ Skipped (already done): {source_url}")

        except Exception as e:
            error_msg = str(e)
            asyncio.run(_update(item_id, CrawlStatus.FAILED, error_msg))
            results.append({"url": source_url, "status": "failed", "error": error_msg})
            log.error(f"[run_etl] ✗ Exception: {source_url} — {error_msg}")

    ti.xcom_push(key="results", value=results)


def send_summary(**context) -> None:
    """Task 3: Log crawl batch summary. Extend with email/Slack when ready.

    Summary format:
        Batch done — 8 done, 1 failed, 1 skipped
        Failed: [url1, ...]
    """
    ti      = context["ti"]
    results = ti.xcom_pull(key="results") or []

    done    = [r for r in results if r["status"] == "done"]
    failed  = [r for r in results if r["status"] == "failed"]
    skipped = [r for r in results if r["status"] == "skipped"]

    summary = (
        f"Crawl batch complete — "
        f"{len(done)} done, {len(failed)} failed, {len(skipped)} skipped "
        f"(total: {len(results)})"
    )
    log.info(f"\n{'='*60}\n{summary}\n{'='*60}")

    if failed:
        for f in failed:
            log.warning(f"  FAILED: {f['url']} — {f.get('error', 'unknown')}")

    # TODO: send email via Airflow EmailOperator when SMTP is configured
    # Airflow will send on_failure_callback email automatically if email_on_failure=True
