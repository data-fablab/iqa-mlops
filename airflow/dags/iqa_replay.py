"""IQA replay DAG skeleton."""

from __future__ import annotations

from datetime import datetime

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
        start_date=datetime(2026, 1, 1),
        tags=["iqa", "replay"],
        params={
            "scenario_id": "production_replay_natural",
            "plan": "data/metadata/casting_flux_replay_plan_natural.csv",
        },
    ) as dag:
        BashOperator(
            task_id="run_replay",
            bash_command=(
                "iqa-run-replay "
                "--scenario-id '{{ params.scenario_id }}' "
                "--plan '{{ params.plan }}'"
            ),
        )
