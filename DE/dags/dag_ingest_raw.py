"""DAG 1: dag_ingest_raw — download audio URLs and stage to MongoDB.

Schedule: every 6 hours.
At completion, triggers dag_process_audio via TriggerDagRunOperator
so DAG 2 always runs immediately after DAG 1 finishes — no fragile time offsets.

Architecture:
    This file contains Airflow wiring only.
    All callable logic lives in dags/wrapper/ingest_raw_wrapper.py.

Pipeline:
    pick_urls → download_and_stage → send_summary → trigger_dag_process
"""
from __future__ import annotations
from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

import sys
sys.path.insert(0, "/opt/airflow")

from dags.wrapper.ingest_raw_wrapper import (
    pick_urls, download_and_stage, send_summary, on_task_failure,
)

default_args = {
    "owner": "chordsense",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "on_failure_callback": on_task_failure,
}

with DAG(
    dag_id="dag_ingest_raw",
    default_args=default_args,
    description="DAG 1: Download audio URLs → stage WAV + metadata to MongoDB, then trigger DAG 2",
    schedule="0 */2 * * *",   # every 6 hours
    start_date=pendulum.datetime(2026, 9, 9, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["chordsense", "etl", "ingest"],
) as dag:

    t1 = PythonOperator(task_id="pick_urls",          python_callable=pick_urls)
    t2 = PythonOperator(task_id="download_and_stage", python_callable=download_and_stage)
    t3 = PythonOperator(task_id="send_summary",       python_callable=send_summary)
    t4 = TriggerDagRunOperator(
        task_id="trigger_dag_process",
        trigger_dag_id="dag_process_audio",
        wait_for_completion=False,   # fire-and-forget — DAG 1 finishes immediately
        reset_dag_run=True,          # allow re-trigger even if previous run exists
    )

    t1 >> t2 >> t3 >> t4
