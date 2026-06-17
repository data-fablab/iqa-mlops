"""IQA Feature-AE lifecycle DAG.

Pipeline stages:
  lifecycle_decision → dataset → train → eval → gates → mlflow → promotion → reload

Issue 08 containerises the first two stages (``lifecycle_decision`` and
``dataset``) via the operator factory (data image), so the Airflow scheduler no
longer imports ``iqa`` for them (ADR 0008). Runtime params (scenario_id, trigger
thresholds, manifest) are passed as templated argv -- no shell, no quoting.

The tail (``train`` … ``reload``) stays on ``PythonOperator`` placeholders until
issues 09-11 containerise it. Real dataset materialisation in MinIO/PostgreSQL is
tracked separately (issue 19), mirroring the ingestion split (issue 18): this
slice only converts the orchestration.
"""

from __future__ import annotations

import os
from datetime import datetime

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
except ImportError:  # pragma: no cover
    DAG = None
    PythonOperator = None

try:
    from iqa.dags.operators import make_container_task
except ImportError:  # pragma: no cover - iqa package absent from the Airflow image.
    make_container_task = None

try:
    from iqa.dags.lifecycle_tasks import (
        task_promotion,
        task_reload,
    )
except ImportError:  # pragma: no cover
    def _placeholder_task(**_context):
        return {"status": "placeholder", "reason": "iqa package not available in airflow image"}

    task_promotion = _placeholder_task
    task_reload = _placeholder_task


DATA_IMAGE = os.environ.get("IQA_IMAGE_DATA", "iqa-data:local")
ML_IMAGE = os.environ.get("IQA_IMAGE_ML", "iqa-ml:local")
GPU_POOL = "iqa_gpu"


dag = None
if (
    DAG is not None
    and PythonOperator is not None
    and make_container_task is not None
    and all([task_promotion, task_reload])
):
    try:
        with DAG(
            dag_id="iqa_lifecycle",
            schedule=None,
            catchup=False,
            start_date=datetime(2026, 1, 1),
            tags=["iqa", "lifecycle"],
            params={
                "regime": "natural",
                "scenario_id": "production_replay_natural",
                "conforming_validated_count": 0,
                "drift_confirmed": False,
                "roi_fail_rate": 0.0,
                "target_stage": "test",
                "manifest": "data/model_datasets/feature_ae_good_v002.csv",
                "candidate_version": "",
                "checkpoint": "models/feature_ae/candidate.pt",
                "validation_set_id": "validation_set_v001",
                "candidate_recall": 1.0,
                "candidate_ap": 0.0,
                "candidate_orange_rate": 0.0,
                "candidate_latency_ms": 0.0,
                "gates_config": "configs/promotion_gates.yaml",
                "run_id": "",
                "registry_stage": "candidate",
                "image": DATA_IMAGE,
                "ml_image": ML_IMAGE,
            },
        ) as _lifecycle_dag:
            op_lifecycle_decision = make_container_task(
                task_id="lifecycle_decision",
                image="{{ params.image }}",
                command=[
                    "iqa-run-lifecycle-decision",
                    "--scenario-id", "{{ params.scenario_id }}",
                    "--conforming-validated-count", "{{ params.conforming_validated_count }}",
                    "--drift-confirmed", "{{ params.drift_confirmed }}",
                    "--roi-fail-rate", "{{ params.roi_fail_rate }}",
                ],
            )

            op_dataset = make_container_task(
                task_id="dataset",
                image="{{ params.image }}",
                command=[
                    "iqa-run-dataset",
                    "--manifest", "{{ params.manifest }}",
                    "--scenario-id", "{{ params.scenario_id }}",
                    "--candidate-version", "{{ params.candidate_version }}",
                ],
            )

            op_train = make_container_task(
                task_id="train",
                image="{{ params.ml_image }}",
                command=[
                    "iqa-run-train",
                    "--scenario-id", "{{ params.scenario_id }}",
                    "--dataset-version", "{{ params.candidate_version }}",
                    "--output-checkpoint", "{{ params.checkpoint }}",
                    "--wait-for-gpu",
                ],
                pool=GPU_POOL,
                gpu_lock=True,
            )

            op_eval = make_container_task(
                task_id="eval",
                image="{{ params.ml_image }}",
                command=[
                    "iqa-run-eval",
                    "--scenario-id", "{{ params.scenario_id }}",
                    "--checkpoint", "{{ params.checkpoint }}",
                    "--validation-set-id", "{{ params.validation_set_id }}",
                    "--wait-for-gpu",
                ],
                pool=GPU_POOL,
                gpu_lock=True,
            )

            op_gates = make_container_task(
                task_id="gates",
                image="{{ params.image }}",
                command=[
                    "iqa-run-gates",
                    "--scenario-id", "{{ params.scenario_id }}",
                    "--recall", "{{ params.candidate_recall }}",
                    "--ap", "{{ params.candidate_ap }}",
                    "--orange-rate", "{{ params.candidate_orange_rate }}",
                    "--latency-ms", "{{ params.candidate_latency_ms }}",
                    "--gates-config", "{{ params.gates_config }}",
                ],
            )

            op_mlflow = make_container_task(
                task_id="mlflow",
                image="{{ params.ml_image }}",
                command=[
                    "iqa-run-mlflow",
                    "--scenario-id", "{{ params.scenario_id }}",
                    "--run-id", "{{ params.run_id }}",
                    "--stage", "{{ params.registry_stage }}",
                ],
            )

            op_promotion = PythonOperator(
                task_id="promotion",
                python_callable=task_promotion,
                doc="Promote model to production",
            )

            op_reload = PythonOperator(
                task_id="reload",
                python_callable=task_reload,
                doc="Reload model in inference service",
            )

            # Linear chain: lifecycle_decision -> dataset -> train -> eval -> gates -> mlflow -> promotion -> reload
            op_lifecycle_decision >> op_dataset >> op_train >> op_eval >> op_gates >> op_mlflow >> op_promotion >> op_reload
        dag = _lifecycle_dag
    except ImportError:  # pragma: no cover - Docker/K8s provider absent (e.g. CI).
        dag = None
