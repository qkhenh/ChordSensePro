"""DAG: every night analytics rollup — chord_attempts → user_chord_mastery.

Architecture:
    This file contains Airflow wiring only (DAG, tasks, dependencies).
    All callable logic lives in dags/wrapper/analytics_rollup_wrapper.py.
    src/ does not know Airflow exists.
"""
from __future__ import annotations
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

import sys
sys.path.insert(0, "/opt/airflow")   # Airflow container root

from dags.wrapper.analytics_rollup_wrapper import (
    aggregate_attempts,
    rolling_accuracy,
    update_mastery,
)

# ── Default args ──────────────────────────────────────────────────────────────

DEFAULT_ARGS = {
    "owner": "qkhenh",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

dag = DAG(
    dag_id="dag_analytics_rollup",
    description="DAG: every night analytics rollup — chord_attempts → user_chord_mastery",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 9, 8),
    schedule="30 23 * * *",
    catchup=False,
    tags=["chordsense", "analytics"],
)

# ── Tasks ─────────────────────────────────────────────────────────────────────

t1 = PythonOperator(task_id="aggregate_attempts", python_callable=aggregate_attempts, dag=dag)
t2 = PythonOperator(task_id="rolling_accuracy",   python_callable=rolling_accuracy,   dag=dag)
t3 = PythonOperator(task_id="update_mastery",     python_callable=update_mastery,     dag=dag)

t1 >> t2 >> t3