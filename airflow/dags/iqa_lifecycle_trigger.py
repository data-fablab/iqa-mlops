"""IQA event-driven lifecycle trigger DAG (ADR 0002, issue 16).

This is the capstone of the orchestration chain: it closes the loop so the
``iqa_lifecycle`` pipeline starts on a *data event* (drift confirmed, full batch,
or enough oracle-validated conformes) without any manual trigger.

Three steps, every metier decision in a container, the trigger itself native
Airflow glue (no ``iqa`` import in the scheduler, ADR 0008):

1. ``evaluate_decision`` -- runs the ``data`` image with ``iqa-run-monitoring``
   via the operator factory. The monitoring rule is evaluated **inside the
   container**; its JSON decision is the task XCom and carries
   ``trigger_lifecycle``.
2. ``gate_on_decision`` -- a ``ShortCircuitOperator``: pure orchestration glue
   (``json`` only, never imports ``iqa``). It reads the container decision and
   short-circuits unless ``trigger_lifecycle`` is true -- so nothing fires on the
   nominal "keep waiting" path.
3. ``trigger_lifecycle`` -- a ``TriggerDagRunOperator`` that launches
   ``iqa_lifecycle`` and forwards the raw signal plus the application lifecycle
   parameters as ``conf``.

The thresholds are the existing data-event rule (configurable via params /
``min_natural_conforming``); the real observation of the store state
(PostgreSQL events / monitoring) is the data plane, tracked by the runtime
sisters (18 / 23). This slice wires the automatic trigger, not the polling I/O.
"""

from __future__ import annotations

import json

from iqa.dags import build_container_dag, data_image, make_container_task, ml_image


DECISION_TASK_ID = "evaluate_decision"
LIFECYCLE_DAG_ID = "iqa_lifecycle"


def _should_trigger(ti=None, **_context) -> bool:
    """Short-circuit glue: trigger the lifecycle iff the container said so.

    Reads the ``evaluate_decision`` container stdout (its JSON decision pushed to
    XCom) and returns ``trigger_lifecycle``. No ``iqa`` import: the rule already
    ran in the container; this only parses the boolean (ADR 0008).
    """
    payload = ti.xcom_pull(task_ids=DECISION_TASK_ID)
    if payload is None:
        return False
    if isinstance(payload, (str, bytes)):
        payload = json.loads(payload)
    return bool(payload.get("trigger_lifecycle", False))


def _define() -> None:
    from airflow.operators.python import ShortCircuitOperator
    from airflow.operators.trigger_dagrun import TriggerDagRunOperator

    op_evaluate_decision = make_container_task(
        task_id=DECISION_TASK_ID,
        image="{{ params.image }}",
        command=[
            "iqa-run-monitoring",
            "--scenario-id", "{{ params.scenario_id }}",
            "--conforming-validated-count", "{{ params.conforming_validated_count }}",
            "--drift-confirmed", "{{ params.drift_confirmed }}",
            "--roi-fail-rate", "{{ params.roi_fail_rate }}",
            "--source-domain", "{{ params.source_domain }}",
            "--window-events", "{{ params.window_events }}",
            "--domain-ratio", "{{ params.domain_ratio }}",
            "--alert-rate", "{{ params.alert_rate }}",
            "--red-rate", "{{ params.red_rate }}",
            "--unexpected-red-rate", "{{ params.unexpected_red_rate }}",
            "--oracle-fn-rate", "{{ params.oracle_fn_rate }}",
            "--critical-window-count", "{{ params.critical_window_count }}",
            "--api-url", "{{ params.api_url }}",
            "--thresholds-config", "{{ params.thresholds_config }}",
        ],
    )

    op_gate_on_decision = ShortCircuitOperator(
        task_id="gate_on_decision",
        python_callable=_should_trigger,
    )

    op_trigger_lifecycle = TriggerDagRunOperator(
        task_id="trigger_lifecycle",
        trigger_dag_id=LIFECYCLE_DAG_ID,
        # Relay the raw signal; iqa_lifecycle re-derives the candidate
        # dataset version in its own lifecycle_decision task.
        conf={
            "scenario_id": "{{ params.scenario_id }}",
            "conforming_validated_count": "{{ params.conforming_validated_count }}",
            "drift_confirmed": "{{ params.drift_confirmed }}",
            "roi_fail_rate": "{{ params.roi_fail_rate }}",
            "image_root": "{{ params.image_root }}",
            "mode": "{{ params.mode }}",
            "max_events": "{{ params.max_events }}",
            "lifecycle_interval": "{{ params.lifecycle_interval }}",
            "max_cycles": "{{ params.max_cycles }}",
            "epochs": "{{ params.epochs }}",
            "max_steps": "{{ params.max_steps }}",
            "gate_eval_profile": "{{ params.gate_eval_profile }}",
            "target_stage": "{{ params.target_stage }}",
            "promotion_min_delta": "{{ params.promotion_min_delta }}",
            "dual_promotion": "{{ params.dual_promotion }}",
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
            "candidate_init_policy": "{{ params.candidate_init_policy }}",
            "external_drift_confirmed": "{{ params.drift_confirmed }}",
            "initial_classification_registered_model": "{{ params.initial_classification_registered_model }}",
            "initial_localization_registered_model": "{{ params.initial_localization_registered_model }}",
            "require_mlflow_registry": "{{ params.require_mlflow_registry }}",
            "mlflow_tracking_uri": "{{ params.mlflow_tracking_uri }}",
            "mlflow_s3_endpoint_url": "{{ params.mlflow_s3_endpoint_url }}",
            "s3_endpoint_url": "{{ params.s3_endpoint_url }}",
            "ml_image": "{{ params.ml_image }}",
        },
    )

    op_evaluate_decision >> op_gate_on_decision >> op_trigger_lifecycle


dag = build_container_dag(
    dag_id="iqa_lifecycle_trigger",
    define=_define,
    schedule="@hourly",
    tags=["iqa", "lifecycle", "trigger"],
    params={
        "scenario_id": "production_replay_natural_piece_b_to_piece_a_p4_drift",
        "conforming_validated_count": 0,
        "drift_confirmed": False,
        "roi_fail_rate": 0.0,
        "source_domain": "piece_a_p4",
        "window_events": 0,
        "domain_ratio": 0.0,
        "alert_rate": 0.0,
        "red_rate": 0.0,
        "unexpected_red_rate": 0.0,
        "oracle_fn_rate": 0.0,
        "critical_window_count": 0,
        "api_url": "http://iqa-api:8000",
        "thresholds_config": "configs/monitoring_thresholds.yaml",
        "image_root": "/opt/iqa/iqa-mlops/.cache/iqa/source_datasets/hss-iad",
        "mode": "progressive-train",
        "max_events": 423,
        "lifecycle_interval": 50,
        "max_cycles": None,
        "epochs": 16,
        "max_steps": None,
        "gate_eval_profile": "full",
        "target_stage": "test",
        "promotion_min_delta": 0.0,
        "dual_promotion": True,
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
        "candidate_init_policy": "active",
        "initial_classification_registered_model": "feature_ae_classifier__production_replay_natural_piece_b_full",
        "initial_localization_registered_model": "feature_ae_localization__production_replay_natural_piece_b_full",
        "require_mlflow_registry": True,
        "mlflow_tracking_uri": "http://mlflow:5000",
        "mlflow_s3_endpoint_url": "http://minio:9000",
        "s3_endpoint_url": "http://minio:9000",
        "ml_image": ml_image(),
        "image": data_image(),
    },
)
