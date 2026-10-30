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
    "on_failure_callback": on_task_failure,
}

with DAG(
    dag_id="dag_process_audio",
    default_args=default_args,
    description="DAG 2: Process staged WAVs → training_sections + training_chords (v2)",
    schedule=None,
    start_date=pendulum.datetime(2026, 9, 9, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["chordsense", "etl", "process"],
) as dag:

    t1 = PythonOperator(task_id="pick_pending_jobs", python_callable=pick_pending_jobs)
    t2 = PythonOperator(task_id="process_and_save",  python_callable=process_and_save)
    t3 = PythonOperator(task_id="send_summary",      python_callable=send_summary)

    t1 >> t2 >> t3
