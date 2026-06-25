"""IQA model rollback DAG (Issue 5).

Execution half of the metric-regression rollback chain: triggered by
``iqa_rollback_sensor`` when the ``IqaModelRegression`` alert fires, it runs the
**existing** rollback path (``iqa.promotion.rollback`` via ``iqa-run-rollback``)
to restore ``previous_prod`` to prod and archive the faulty version, then asks
``iqa-inference`` to reload so production serves the restored model.

Rollback is a data-plane operation (MLflow Registry only, no GPU/torch): it runs
on the data image and never holds the GPU lock. The scheduler imports only the
lightweight DAG factory and never the IQA runtime (ADR 0008).
"""

from __future__ import annotations

from datetime import timedelta

from iqa.dags import build_container_dag, data_image, make_container_task


def _define() -> None:
    op_rollback = make_container_task(
        task_id="run_rollback",
        image="{{ params.data_image }}",
        command=(
            "iqa-run-rollback "
            "--scenario-id {{ params.scenario_id }}"
            "{% if params.faulty_version %} --faulty-version {{ params.faulty_version }}{% endif %}"
        ),
        env={
            "MLFLOW_TRACKING_URI": "{{ params.mlflow_tracking_uri }}",
            "PYTHONPATH": "{{ params.repo_root }}:{{ params.repo_root }}/src",
        },
        working_dir="/opt/iqa/iqa-mlops",
        retries=0,
        execution_timeout=timedelta(minutes=15),
    )

    op_reload = make_container_task(
        task_id="run_reload",
        image="{{ params.data_image }}",
        command=(
            "iqa-run-reload "
            "--scenario-id {{ params.scenario_id }} "
            "--target-stage prod"
        ),
        env={
            "MLFLOW_TRACKING_URI": "{{ params.mlflow_tracking_uri }}",
            "PYTHONPATH": "{{ params.repo_root }}:{{ params.repo_root }}/src",
        },
        working_dir="/opt/iqa/iqa-mlops",
        retries=0,
        execution_timeout=timedelta(minutes=15),
    )

    op_rollback >> op_reload


dag = build_container_dag(
    dag_id="iqa_rollback",
    define=_define,
    schedule=None,
    tags=["iqa", "rollback", "promotion"],
    max_active_runs=1,
    catchup=False,
    params={
        "scenario_id": "production_replay_natural",
        # Empty -> the CLI rolls back from the current prod version.
        "faulty_version": "",
        "repo_root": "/opt/iqa/iqa-mlops",
        "mlflow_tracking_uri": "http://mlflow:5000",
        "data_image": data_image(),
    },
)
