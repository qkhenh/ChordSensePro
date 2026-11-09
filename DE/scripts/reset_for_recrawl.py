"""Reset the crawl state so the whole pipeline can be re-run with the new code.

Why: before the DE refactor, JAAH jobs were staged without their annotation, so their Mongo
documents / crawl_queue rows are "done" but carry no labels. Mongo's ``exists()`` dedup would also
block re-ingesting the same URLs. This script wipes the staging area and re-queues the sources.

What it does
  1. MongoDB  raw_audio_jobs      → delete ALL documents (staging only; pointers to temp WAVs)
  2. Postgres crawl_queue         → status done/failed/skipped/running  →  pending (error cleared)
  3. (optional --truncate-samples) Postgres training_samples → TRUNCATE   (+ remove /data/training/audio)
  Never touches: song_analyses (parked old-pipeline data), dataset files, annotations.

Run inside the airflow container:
    docker exec chordsense_airflow python /opt/airflow/scripts/reset_for_recrawl.py --dry-run
    docker exec chordsense_airflow python /opt/airflow/scripts/reset_for_recrawl.py
    docker exec chordsense_airflow python /opt/airflow/scripts/reset_for_recrawl.py --only-failed   # keep done rows
  docker exec chordsense_airflow python /opt/airflow/scripts/reset_for_recrawl.py --source jaah --truncate-samples --priority 8

Options
  --dry-run            print what would change, change nothing
  --only-failed        only re-queue failed/running/skipped rows (keep 'done'); Mongo docs of those only
  --truncate-samples   also clear training_samples and the .npy files
  --source TYPE        limit everything to one source_type (e.g. jaah); with --truncate-samples only that
                       source's training_samples rows and the .npy files they reference are removed
  --priority N         set crawl_queue.priority = N on the re-queued rows (so they run before the backlog)
"""
from __future__ import annotations
import argparse
import asyncio
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "/opt/airflow")

from src.shared.infrastructure.mongo.client import get_mongo_db            # noqa: E402
from src.shared.infrastructure.postgres.client import get_connection       # noqa: E402
from src.data_processing.application.pipeline_handlers.sample_writer_handler import training_dir  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only-failed", action="store_true")
    ap.add_argument("--truncate-samples", action="store_true")
    ap.add_argument("--source", default=None)
    ap.add_argument("--priority", type=int, default=None)
    args = ap.parse_args()
    dry = args.dry_run
    source = args.source

    statuses = ["failed", "running", "skipped"] if args.only_failed else ["done", "failed", "running", "skipped"]
    src_sql = " AND source_type = %s" if source else ""
    src_args = (source,) if source else ()

    # ── Postgres: crawl_queue ────────────────────────────────────────────────
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT status, COUNT(*) FROM crawl_queue GROUP BY status ORDER BY status")
        print("crawl_queue before :", dict(cur.fetchall()))
        cur.execute(f"SELECT source_url FROM crawl_queue WHERE status = ANY(%s){src_sql}", (statuses, *src_args))
        urls = [r[0] for r in cur.fetchall()]
        print(f"  → {len(urls)} rows will be set back to 'pending' (statuses: {statuses}, source: {source or 'all'})")
        if not dry:
            prio_sql = ", priority = %s" if args.priority is not None else ""
            prio_args = (args.priority,) if args.priority is not None else ()
            cur.execute(
                f"UPDATE crawl_queue SET status='pending', error_message=NULL, processed_at=NULL{prio_sql} "
                f"WHERE status = ANY(%s){src_sql}", (*prio_args, statuses, *src_args),
            )

    # ── Mongo: raw_audio_jobs ────────────────────────────────────────────────
    async def reset_mongo() -> None:
        col = get_mongo_db()["raw_audio_jobs"]
        total = await col.count_documents({})
        query = {"source_url": {"$in": urls}} if (args.only_failed or source) else {}
        n = await col.count_documents(query)
        print(f"mongo raw_audio_jobs: {total} docs, {n} will be deleted")
        if not dry:
            res = await col.delete_many(query)
            print(f"  deleted {res.deleted_count}")

    asyncio.run(reset_mongo())

    # ── Optional: training_samples + npy files ───────────────────────────────
    if args.truncate_samples:
        audio_dir = training_dir() / "audio"
        with get_connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT to_regclass('training_samples')")
            row = cur.fetchone()
            exists = bool(row and row[0] is not None)
            if exists and source:
                cur.execute("SELECT DISTINCT audio_path FROM training_samples WHERE source_type = %s", (source,))
                paths = [r[0] for r in cur.fetchall()]
                cur.execute("SELECT COUNT(*) FROM training_samples WHERE source_type = %s", (source,))
                count_row = cur.fetchone()
                print(f"training_samples[{source}]: {count_row[0] if count_row else 0} rows, {len(paths)} .npy files"
                      f" will be {'(dry) ' if dry else ''}deleted")
                if not dry:
                    cur.execute("DELETE FROM training_samples WHERE source_type = %s", (source,))
                    for rel in paths:
                        (training_dir() / rel).unlink(missing_ok=True)
            elif exists:
                cur.execute("SELECT COUNT(*) FROM training_samples")
                count_row = cur.fetchone()
                print(f"training_samples: {count_row[0] if count_row else 0} rows will be truncated")
                if not dry:
                    cur.execute("TRUNCATE training_samples RESTART IDENTITY")
        if not source:
            print(f"audio dir {audio_dir} {'(would be removed)' if dry else 'removed'}")
            if not dry and audio_dir.exists():
                shutil.rmtree(audio_dir, ignore_errors=True)

    print("\nDRY RUN — nothing changed." if dry else "\nDone. Unpause the DAGs to start crawling.")


if __name__ == "__main__":
    main()
