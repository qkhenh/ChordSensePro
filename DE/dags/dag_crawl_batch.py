"""DAG: auto crawl batch — picks pending URLs from crawl_queue and runs ETL pipeline.

Schedule: every 6 hours. Tune BATCH_SIZE in wrapper to control throughput.

Architecture:
    This file contains Airflow wiring only.
    All callable logic lives in dags/wrapper/crawl_batch_wrapper.py.

Pipeline:
    pick_urls → run_etl → send_summary
"""
from __future__ import annotations
from datetime import datetime, timedelta

import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator

import sys
sys.path.insert(0, "/opt/airflow")

from dags.wrapper.crawl_batch_wrapper import pick_urls, run_etl, send_summary

# ── Default args ──────────────────────────────────────────────────────────────

default_args = {
    "owner": "chordsense",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": True,   # Airflow sends alert email on task failure
}

with DAG(
    dag_id="dag_crawl_batch",
    default_args=default_args,
    description="Auto crawl: pick pending URLs → ETL pipeline → summary",
    schedule="0 */6 * * *",   # every 6 hours
    start_date=pendulum.datetime(2026, 9, 8, tz="UTC"),
    catchup=False,
    max_active_runs=1,         # prevent overlapping batch runs
    tags=["chordsense", "etl", "crawl"],
) as dag:

    t1 = PythonOperator(task_id="pick_urls",    python_callable=pick_urls)
    t2 = PythonOperator(task_id="run_etl",      python_callable=run_etl)
    t3 = PythonOperator(task_id="send_summary", python_callable=send_summary)

    t1 >> t2 >> t3
