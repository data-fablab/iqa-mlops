"""IQA Feature-AE application lifecycle DAG.

Airflow orchestrates the application lifecycle as a containerised workflow. This
is the pipeline applicatif Feature-AE de reference. The metier logic lives in
``iqa-run-replay-lifecycle-cycle``: replay window, progressive training, fair
active-vs-candidate evaluation, MLflow evidence and test-stage promotion. The
scheduler imports only the lightweight DAG factory and never imports the IQA
runtime (ADR 0008).
"""

from __future__ import annotations

from datetime import timedelta

from iqa.dags import build_container_dag, make_container_task, ml_image

GPU_POOL = "iqa_gpu"


def _define() -> None:
    make_container_task(
        task_id="run_application_lifecycle",
        image="{{ params.ml_image }}",
        command=(
            "iqa-run-replay-lifecycle-cycle "
            "--scenario-id {{ params.scenario_id }} "
            "--image-root {{ params.repo_root }}/data/raw/hss-iad "
            "--mode {{ params.mode }} "
            "--max-events {{ params.max_events }} "
            "--lifecycle-interval {{ params.lifecycle_interval }} "
            "--max-cycles {{ params.max_cycles }} "
            "--epochs {{ params.epochs }} "
            "--target-stage {{ params.target_stage }} "
            "--promotion-min-delta {{ params.promotion_min_delta }} "
            "--publish-minio "
            "--wait-for-gpu"
            "{% if params.require_mlflow_registry %} --require-mlflow-registry{% endif %}"
        ),
        pool=GPU_POOL,
        gpu_lock=True,
        repo_mount=True,
        working_dir="/opt/iqa/iqa-mlops",
        retries=0,
        execution_timeout=timedelta(hours=6),
    )


dag = build_container_dag(
    dag_id="iqa_lifecycle",
    define=_define,
    schedule=None,
    tags=["iqa", "lifecycle", "feature-ae"],
    max_active_runs=1,
    catchup=False,
    params={
        "scenario_id": "production_replay_natural",
        "repo_root": "/opt/iqa/iqa-mlops",
        "mode": "progressive-train",
        "max_events": 260,
        "lifecycle_interval": 50,
        "max_cycles": 3,
        "epochs": 10,
        "target_stage": "test",
        "promotion_min_delta": 0.0,
        "require_mlflow_registry": False,
        "ml_image": ml_image(),
    },
)
