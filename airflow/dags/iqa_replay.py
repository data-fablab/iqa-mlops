"""IQA replay DAG skeleton."""

from __future__ import annotations

try:
    from airflow import DAG
    from airflow.operators.bash import BashOperator
except ImportError:  # pragma: no cover
    DAG = None
    BashOperator = None


dag = None
if DAG is not None and BashOperator is not None:
    with DAG(
        dag_id="iqa_replay",
        schedule=None,
        catchup=False,
        tags=["iqa", "replay"],
    ) as dag:
        BashOperator(task_id="run_replay", bash_command="iqa-run-replay")
