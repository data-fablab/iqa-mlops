"""IQA Piece B stable -> Piece A/P4 drift observation and correction DAG.

This DAG is the natural drift scenario. It first runs an inference-only replay
with the stable Piece B registry models, then triggers exactly one lifecycle
correction only when the observed drift metrics confirm a real degradation.
"""

from __future__ import annotations

import json
from datetime import timedelta

from iqa.dags import build_container_dag, make_container_task, ml_image


OBSERVE_TASK_ID = "observe_replay"
LIFECYCLE_DAG_ID = "iqa_lifecycle"
GPU_POOL = "iqa_gpu"


def _should_trigger_correction(ti=None, **_context) -> bool:
    payload = ti.xcom_pull(task_ids=OBSERVE_TASK_ID)
    payload = _parse_observation_payload(payload)
    return bool(payload.get("trigger_lifecycle", False))


def _parse_observation_payload(payload) -> dict:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    if isinstance(payload, (list, tuple)):
        payload = "\n".join(str(item) for item in payload)
    if not isinstance(payload, str):
        raise TypeError(f"unsupported drift observation XCom payload type: {type(payload).__name__}")
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        start = payload.find("{")
        end = payload.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(payload[start : end + 1])


def _define() -> None:
    from airflow.operators.python import ShortCircuitOperator
    from airflow.operators.trigger_dagrun import TriggerDagRunOperator

    op_observe_replay = make_container_task(
        task_id=OBSERVE_TASK_ID,
        image="{{ params.ml_image }}",
        command=(
            "iqa-run-drift-observation-replay "
            "--scenario-id {{ params.scenario_id }} "
            "--image-root {{ params.image_root }} "
            "--max-events {{ params.max_events }} "
            "--window-size {{ params.window_size }} "
            "--target-stage {{ params.target_stage }} "
            "--thresholds-config {{ params.thresholds_config }} "
            "--api-url {{ params.api_url }} "
            "--initial-classification-registered-model {{ params.initial_classification_registered_model }} "
            "--initial-localization-registered-model {{ params.initial_localization_registered_model }}"
            "{% if params.require_mlflow_registry in [true, 'True', 'true', '1', 1] %} --require-mlflow-registry{% endif %}"
        ),
        env={
            "MLFLOW_TRACKING_URI": "{{ params.mlflow_tracking_uri }}",
            "IQA_MLFLOW_TRACKING_URI": "{{ params.mlflow_tracking_uri }}",
            "MLFLOW_S3_ENDPOINT_URL": "{{ params.mlflow_s3_endpoint_url }}",
            "IQA_S3_ENDPOINT_URL": "{{ params.s3_endpoint_url }}",
            "PYTHONPATH": "{{ params.repo_root }}:{{ params.repo_root }}/src",
        },
        pool=GPU_POOL,
        gpu_lock=True,
        repo_mount=True,
        working_dir="/opt/iqa/iqa-mlops",
        retries=0,
        execution_timeout=timedelta(hours=4),
    )

    op_gate_on_confirmed_drift = ShortCircuitOperator(
        task_id="gate_on_confirmed_drift",
        python_callable=_should_trigger_correction,
    )

    op_trigger_correction = TriggerDagRunOperator(
        task_id="trigger_lifecycle_correction",
        trigger_dag_id=LIFECYCLE_DAG_ID,
        conf={
            "scenario_id": "{{ params.scenario_id }}",
            "repo_root": "{{ params.repo_root }}",
            "image_root": "{{ params.image_root }}",
            "mode": "progressive-train",
            "max_events": "{{ params.max_events }}",
            "lifecycle_interval": "{{ params.lifecycle_interval }}",
            "max_cycles": 1,
            "epochs": "{{ params.epochs }}",
            "max_steps": "{{ params.max_steps }}",
            "gate_eval_profile": "{{ params.gate_eval_profile }}",
            "target_stage": "{{ params.target_stage }}",
            "promotion_min_delta": "{{ params.promotion_min_delta }}",
            "dual_promotion": True,
            "localization_promotion_min_delta": "{{ params.localization_promotion_min_delta }}",
            "classification_require_fn_improvement": "{{ params.classification_require_fn_improvement }}",
            "classification_min_image_recall_delta": "{{ params.classification_min_image_recall_delta }}",
            "classification_min_image_ap_delta": "{{ params.classification_min_image_ap_delta }}",
            "anchor_good_manifest": "{{ params.anchor_good_manifest }}",
            "anchor_good_max_per_class": "{{ params.anchor_good_max_per_class }}",
            "reference_eval_manifest": "{{ params.reference_eval_manifest }}",
            "classification_selection_manifest": "{{ params.classification_selection_manifest }}",
            "reference_gt_masks_manifest": "{{ params.reference_gt_masks_manifest }}",
            "max_good_red_regression": "{{ params.max_good_red_regression }}",
            "candidate_init_policy": "active",
            "external_drift_confirmed": True,
            "initial_classification_registered_model": "{{ params.initial_classification_registered_model }}",
            "initial_localization_registered_model": "{{ params.initial_localization_registered_model }}",
            "require_mlflow_registry": "{{ params.require_mlflow_registry }}",
            "mlflow_tracking_uri": "{{ params.mlflow_tracking_uri }}",
            "mlflow_s3_endpoint_url": "{{ params.mlflow_s3_endpoint_url }}",
            "s3_endpoint_url": "{{ params.s3_endpoint_url }}",
            "ml_image": "{{ params.ml_image }}",
        },
    )

    op_observe_replay >> op_gate_on_confirmed_drift >> op_trigger_correction


dag = build_container_dag(
    dag_id="iqa_drift_piece_a_p4",
    define=_define,
    schedule=None,
    tags=["iqa", "drift", "piece-a-p4"],
    max_active_runs=1,
    catchup=False,
    params={
        "scenario_id": "production_replay_natural_piece_b_to_piece_a_p4_drift",
        "repo_root": "/opt/iqa/iqa-mlops",
        "image_root": "/opt/iqa/iqa-mlops/.cache/iqa/source_datasets/hss-iad",
        "max_events": 423,
        "window_size": 30,
        "thresholds_config": "configs/monitoring_thresholds.yaml",
        "api_url": "http://iqa-api:8000",
        "lifecycle_interval": 50,
        "epochs": 16,
        "max_steps": None,
        "gate_eval_profile": "full",
        "target_stage": "test",
        "promotion_min_delta": 0.0,
        "localization_promotion_min_delta": 0.0,
        "classification_require_fn_improvement": True,
        "classification_min_image_recall_delta": 0.0,
        "classification_min_image_ap_delta": 0.0,
        "anchor_good_manifest": "data/metadata/feature_ae_bootstrap_piece_b_minimal_v001.csv",
        "anchor_good_max_per_class": 256,
        "reference_eval_manifest": "data/validation/validation_set_piece_b_to_piece_a_p4_drift_v001.csv",
        "classification_selection_manifest": "data/validation/classification_selection_piece_b_to_piece_a_p4_drift_v001.csv",
        "reference_gt_masks_manifest": "data/validation/validation_gt_masks_piece_b_to_piece_a_p4_drift_v001.csv",
        "max_good_red_regression": 1,
        "initial_classification_registered_model": "feature_ae_classifier__production_replay_natural_piece_b_full",
        "initial_localization_registered_model": "feature_ae_localization__production_replay_natural_piece_b_full",
        "require_mlflow_registry": True,
        "mlflow_tracking_uri": "http://mlflow:5000",
        "mlflow_s3_endpoint_url": "http://minio:9000",
        "s3_endpoint_url": "http://minio:9000",
        "ml_image": ml_image(),
    },
)
