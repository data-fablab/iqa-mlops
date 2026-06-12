"""IQA ingestion DAG skeleton."""

from __future__ import annotations

try:
    from airflow import DAG
    from airflow.operators.bash import BashOperator
except ImportError:  # pragma: no cover - lets CI import the module without Airflow extras.
    DAG = None
    BashOperator = None


dag = None
if DAG is not None and BashOperator is not None:
    with DAG(
        dag_id="iqa_ingestion",
        schedule=None,
        catchup=False,
        tags=["iqa", "ingestion"],
    ) as dag:
        BashOperator(task_id="run_ingestion", bash_command="iqa-run-ingestion")
