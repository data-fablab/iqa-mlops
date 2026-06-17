"""IQA monitoring DAG skeleton."""

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
        dag_id="iqa_monitoring",
        schedule="@hourly",
        catchup=False,
        start_date=datetime(2026, 1, 1),
        tags=["iqa", "monitoring"],
        params={
            "scenario_id": "production_replay_natural",
            "conforming_validated_count": 0,
            "drift_confirmed": False,
            "roi_fail_rate": 0.0,
        },
    ) as dag:
        BashOperator(
            task_id="evaluate_lifecycle_conditions",
            bash_command=(
                "iqa-run-monitoring "
                "--scenario-id '{{ params.scenario_id }}' "
                "--conforming-validated-count '{{ params.conforming_validated_count }}' "
                "{% if params.drift_confirmed %}--drift-confirmed {% endif %}"
                "--roi-fail-rate '{{ params.roi_fail_rate }}'"
            ),
        )
