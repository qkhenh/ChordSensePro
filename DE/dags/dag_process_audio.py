"""DAG 2: dag_process_audio — process staged WAVs from MongoDB → save to PostgreSQL.

Schedule: every 6 hours, offset +30 min from dag_ingest_raw so DAG 1 finishes first.
Picks pending RawAudioJobs from MongoDB → Demucs → beats → features → PostgreSQL.

Architecture:
    This file contains Airflow wiring only.
    All callable logic lives in dags/wrapper/process_audio_wrapper.py.

Pipeline:
    pick_pending_jobs → process_and_save → send_summary

Dependency:
    Runs 30 min after dag_ingest_raw (schedule offset).
    If MongoDB has no pending jobs, tasks complete immediately — no-op.
"""
from __future__ import annotations
from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator

import sys
sys.path.insert(0, "/opt/airflow")

from dags.wrapper.process_audio_wrapper import (
    pick_pending_jobs, process_and_save, send_summary, on_task_failure,
)

default_args = {
    "owner": "chordsense",
    "retries": 1,
    "retry_delay": timedelta(minutes=15),
    "on_failure_callback": on_task_failure,   # email on any task failure
}

with DAG(
    dag_id="dag_process_audio",
    default_args=default_args,
    description="DAG 2: Process staged WAVs (Demucs + features) → save to PostgreSQL",
    schedule=None,   # triggered by dag_ingest_raw via TriggerDagRunOperator — not time-based
    start_date=pendulum.datetime(2026, 9, 9, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["chordsense", "etl", "process"],
) as dag:

    t1 = PythonOperator(task_id="pick_pending_jobs", python_callable=pick_pending_jobs)
    t2 = PythonOperator(task_id="process_and_save",  python_callable=process_and_save)
    t3 = PythonOperator(task_id="send_summary",      python_callable=send_summary)

    t1 >> t2 >> t3
