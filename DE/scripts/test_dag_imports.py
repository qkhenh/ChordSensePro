"""Test DAG imports to find any broken imports."""
import sys
sys.path.insert(0, "/opt/airflow")

for dag_name in ["dag_ingest_raw", "dag_process_audio"]:
    try:
        __import__(f"dags.{dag_name}")
        print(f"OK: {dag_name}")
    except Exception as e:
        print(f"FAIL: {dag_name}: {e}")
