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
CORRECTION_DAG_ID = "iqa_drift_correction_lifecycle"
CORRECTION_CONF_TASK_ID = "build_correction_conf"
GPU_POOL = "iqa_gpu"


def _should_trigger_correction(ti=None, **_context) -> bool:
    payload = ti.xcom_pull(task_ids=OBSERVE_TASK_ID)
    payload = _parse_observation_payload(payload)
    return bool(payload.get("trigger_lifecycle", False))


def _build_correction_conf(ti=None, **_context) -> dict:
    payload = _parse_observation_payload(ti.xcom_pull(task_ids=OBSERVE_TASK_ID))
    return {
        "drift_context_path": payload.get("drift_context_path", ""),
        "correction_anchor_manifest_path": payload.get("correction_anchor_manifest_path", ""),
        "balanced_eval_manifest_path": payload.get("balanced_eval_manifest_path", ""),
        "observation_run_id": payload.get("run_id", ""),
        "scenario_id": payload.get("scenario_id", ""),
        "context_events_total": payload.get("context_events_total", 0),
        "lifecycle_interval": payload.get("context_events_total", 0),
        "manifest_balance": payload.get("manifest_balance", {}),
    }


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
    from airflow.operators.python import PythonOperator, ShortCircuitOperator
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
            "--stable-reference-events {{ params.stable_reference_events }} "
            "--drift-observation-windows {{ params.drift_observation_windows }} "
            "--target-stage {{ params.target_stage }} "
            "--thresholds-config {{ params.thresholds_config }} "
            "--api-url {{ params.api_url }} "
            "{% if params.initial_classification_registered_model %}"
            "--initial-classification-registered-model {{ params.initial_classification_registered_model }} "
            "{% endif %}"
            "{% if params.initial_localization_registered_model %}"
            "--initial-localization-registered-model {{ params.initial_localization_registered_model }} "
            "{% endif %}"
            "{% if params.initial_classification_registered_version %}"
            "--initial-classification-registered-version {{ params.initial_classification_registered_version }} "
            "{% endif %}"
            "{% if params.initial_localization_registered_version %}"
            "--initial-localization-registered-version {{ params.initial_localization_registered_version }} "
            "{% endif %}"
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

    op_build_correction_conf = PythonOperator(
        task_id=CORRECTION_CONF_TASK_ID,
        python_callable=_build_correction_conf,
    )

    op_trigger_correction = TriggerDagRunOperator(
        task_id="trigger_drift_correction_lifecycle",
        trigger_dag_id=CORRECTION_DAG_ID,
        conf={
            "scenario_id": "{{ params.scenario_id }}",
            "repo_root": "{{ params.repo_root }}",
            "image_root": "{{ params.image_root }}",
            "drift_context_path": "{{ ti.xcom_pull(task_ids='build_correction_conf')['drift_context_path'] }}",
            "anchor_good_manifest": "{{ ti.xcom_pull(task_ids='build_correction_conf')['correction_anchor_manifest_path'] }}",
            "reference_eval_manifest": "{{ ti.xcom_pull(task_ids='build_correction_conf')['balanced_eval_manifest_path'] }}",
            "observation_run_id": "{{ ti.xcom_pull(task_ids='build_correction_conf')['observation_run_id'] }}",
            "context_events_total": "{{ ti.xcom_pull(task_ids='build_correction_conf')['context_events_total'] }}",
            "lifecycle_interval": "{{ ti.xcom_pull(task_ids='build_correction_conf')['lifecycle_interval'] }}",
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
            "anchor_good_max_per_class": "{{ params.anchor_good_max_per_class }}",
            "classification_selection_manifest": "{{ params.classification_selection_manifest }}",
            "reference_gt_masks_manifest": "{{ params.reference_gt_masks_manifest }}",
            "max_good_red_regression": "{{ params.max_good_red_regression }}",
            "candidate_init_policy": "{{ params.candidate_init_policy }}",
            "skip_report_only_reference_eval": "{{ params.skip_report_only_reference_eval }}",
            "external_drift_confirmed": True,
            "initial_classification_registered_model": "{{ params.initial_classification_registered_model }}",
            "initial_localization_registered_model": "{{ params.initial_localization_registered_model }}",
            "initial_classification_registered_version": "{{ params.initial_classification_registered_version }}",
            "initial_localization_registered_version": "{{ params.initial_localization_registered_version }}",
            "require_mlflow_registry": "{{ params.require_mlflow_registry }}",
            "mlflow_tracking_uri": "{{ params.mlflow_tracking_uri }}",
            "mlflow_s3_endpoint_url": "{{ params.mlflow_s3_endpoint_url }}",
            "s3_endpoint_url": "{{ params.s3_endpoint_url }}",
            "api_url": "{{ params.api_url }}",
            "ml_image": "{{ params.ml_image }}",
        },
    )

    op_observe_replay >> op_gate_on_confirmed_drift >> op_build_correction_conf >> op_trigger_correction


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
        "max_events": 40,
        "window_size": 10,
        "stable_reference_events": 10,
        "drift_observation_windows": 3,
        "thresholds_config": "configs/monitoring_thresholds.yaml",
        "api_url": "http://iqa-api:8000",
        "lifecycle_interval": 24,
        "epochs": 4,
        "max_steps": None,
        "gate_eval_profile": "fast",
        "target_stage": "test",
        "promotion_min_delta": 0.0,
        "localization_promotion_min_delta": 0.0,
        "classification_require_fn_improvement": True,
        "classification_min_image_recall_delta": 0.0,
        "classification_min_image_ap_delta": 0.0,
        "anchor_good_manifest": "data/metadata/feature_ae_bootstrap_piece_b_minimal_v001.csv",
        "anchor_good_max_per_class": 256,
        "candidate_init_policy": "stable_base",
        "skip_report_only_reference_eval": True,
        "reference_eval_manifest": "data/validation/validation_set_piece_b_to_piece_a_p4_drift_v001.csv",
        "classification_selection_manifest": "data/validation/classification_selection_piece_b_to_piece_a_p4_drift_v001.csv",
        "reference_gt_masks_manifest": "data/validation/validation_gt_masks_piece_b_to_piece_a_p4_drift_v001.csv",
        "max_good_red_regression": 1,
        "initial_classification_registered_model": "",
        "initial_localization_registered_model": "",
        "initial_classification_registered_version": "",
        "initial_localization_registered_version": "",
        "require_mlflow_registry": False,
        "mlflow_tracking_uri": "http://mlflow:5000",
        "mlflow_s3_endpoint_url": "http://minio:9000",
        "s3_endpoint_url": "http://minio:9000",
        "ml_image": ml_image(),
    },
)
