"""IQA ingestion DAG skeleton."""

from __future__ import annotations

from datetime import datetime

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
        start_date=datetime(2026, 1, 1),
        tags=["iqa", "ingestion"],
        params={
            "manifest": "data/metadata/casting_piece_events.csv",
            "source": "historical_replay",
            "scenario_id": "raw_ingestion",
        },
    ) as dag:
        BashOperator(
            task_id="run_ingestion",
            bash_command=(
                "iqa-run-ingestion "
                "--manifest '{{ params.manifest }}' "
                "--source '{{ params.source }}' "
                "--scenario-id '{{ params.scenario_id }}'"
            ),
        )
