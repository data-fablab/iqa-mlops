"""IQA Feature-AE lifecycle DAG.

Pipeline stages:
  lifecycle_decision → dataset → train → eval → gates → mlflow → promotion → reload

Issue 11 completes the conversion: every stage now runs as a container via the
operator factory, so the Airflow scheduler never imports ``iqa`` (ADR 0008 fully
resolved -- no ``PythonOperator`` left). Runtime params (scenario_id, stages,
manifest) are passed as templated argv -- no shell, no quoting.

The container stdout is each task's XCom (references only, no payloads). The real
runtime behind these boundaries (dataset materialisation, train/eval, MLflow
registration, the Registry transition and the inference HTTP reload) is tracked
by the runtime sister issues 18-22: this slice only converts the orchestration.
"""

from __future__ import annotations

try:
    from iqa.dags import build_container_dag, data_image, make_container_task, ml_image
except ImportError:  # pragma: no cover - iqa package absent from the Airflow image.
    build_container_dag = data_image = make_container_task = ml_image = None

GPU_POOL = "iqa_gpu"


def _define() -> None:
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
            # XCom from the dataset task: the s3:// URI of the materialised
            # candidate (its last stdout line), so train resolves it by URI.
            "--dataset-uri", "{{ ti.xcom_pull(task_ids='dataset') }}",
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

    op_promotion = make_container_task(
        task_id="promotion",
        image="{{ params.ml_image }}",
        command=[
            "iqa-run-promotion",
            "--scenario-id", "{{ params.scenario_id }}",
            "--source-stage", "{{ params.registry_stage }}",
            "--target-stage", "{{ params.target_stage }}",
        ],
    )

    op_reload = make_container_task(
        task_id="reload",
        image="{{ params.image }}",
        command=[
            "iqa-run-reload",
            "--scenario-id", "{{ params.scenario_id }}",
            "--target-stage", "{{ params.target_stage }}",
        ],
    )

    # Linear chain: lifecycle_decision -> dataset -> train -> eval -> gates -> mlflow -> promotion -> reload
    op_lifecycle_decision >> op_dataset >> op_train >> op_eval >> op_gates >> op_mlflow >> op_promotion >> op_reload


dag = (
    build_container_dag(
        dag_id="iqa_lifecycle",
        define=_define,
        schedule=None,
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
            "image": data_image(),
            "ml_image": ml_image(),
        },
    )
    if build_container_dag is not None
    else None
)
