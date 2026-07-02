"""IQA drift-correction lifecycle DAG.

This DAG is launched by ``iqa_drift_piece_a_p4`` after ROI novelty confirms P4
drift. It consumes the drift context manifest plus balanced Piece B/P4 train and
evaluation manifests, so the corrective lifecycle does not replay the full
baseline and does not overfit the weights to P4 only.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from datetime import timedelta

from airflow.models.dag import DAG as AirflowDAG
from iqa.dags import build_container_dag, make_container_task, ml_image

GPU_POOL = "iqa_gpu"

_AIRFLOW_DAG_TYPE = AirflowDAG


def _safe_metric_run_id(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.:-]+", "_", value)
    return value[:180] or "airflow_failed"


def _clear_lifecycle_metrics_on_failure(context) -> None:
    params = context.get("params") or {}
    dag_run = context.get("dag_run")
    run_id = getattr(dag_run, "run_id", "") or context.get("run_id") or "airflow_failed"
    api_url = str(params.get("api_url") or os.environ.get("IQA_API_URL") or "").strip().rstrip("/")
    scenario_id = str(params.get("scenario_id") or "production_replay_natural_piece_b_to_piece_a_p4_drift")
    if not api_url:
        return
    payload = {
        "event_type": "run_failed",
        "scenario_id": scenario_id,
        "lifecycle_run_id": f"airflow_{_safe_metric_run_id(run_id)}",
    }
    data = json.dumps(payload, sort_keys=True).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("IQA_SERVICE_TOKEN", "").strip()
    if token:
        headers["X-IQA-Service-Token"] = token
    request = urllib.request.Request(
        f"{api_url}/internal/lifecycle/events",
        data=data,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            response.read()
    except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
        print(f"warning: lifecycle failure cleanup failed: {type(exc).__name__}: {exc}", flush=True)


def _define() -> None:
    make_container_task(
        task_id="run_drift_correction_lifecycle",
        image="{{ params.ml_image }}",
        command=(
            "iqa-run-replay-lifecycle-cycle "
            "--scenario-id {{ params.scenario_id }} "
            "--image-root {{ params.image_root }} "
            "--mode progressive-train "
            "--drift-context-path {{ params.drift_context_path }} "
            "--lifecycle-interval {{ params.lifecycle_interval }} "
            "--max-cycles 1 "
            "--epochs {{ params.epochs }} "
            "{% if params.max_steps not in [none, 'None', 'none', 'null', ''] %}--max-steps {{ params.max_steps }} {% endif %}"
            "--gate-eval-profile {{ params.gate_eval_profile }} "
            "--target-stage {{ params.target_stage }} "
            "{% if params.promotion_min_delta not in [none, 'None', 'none', 'null', ''] %}"
            "--promotion-min-delta {{ params.promotion_min_delta }} "
            "{% endif %}"
            "--anchor-good-manifest {{ params.anchor_good_manifest }} "
            "--anchor-good-max-per-class {{ params.anchor_good_max_per_class }} "
            "--reference-eval-manifest {{ params.reference_eval_manifest }} "
            "--classification-selection-manifest {{ params.classification_selection_manifest }} "
            "--reference-gt-masks-manifest {{ params.reference_gt_masks_manifest }} "
            "--max-good-red-regression {{ params.max_good_red_regression }} "
            "--candidate-init-policy {{ params.candidate_init_policy }} "
            "{% if params.skip_report_only_reference_eval in [true, 'True', 'true', '1', 1] %}"
            "--skip-report-only-reference-eval "
            "{% endif %}"
            "--external-drift-confirmed "
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
            "--publish-minio "
            "--wait-for-gpu "
            "--dual-promotion "
            "{% if params.localization_promotion_min_delta not in [none, 'None', 'none', 'null', ''] %}"
            "--localization-promotion-min-delta {{ params.localization_promotion_min_delta }} "
            "{% endif %}"
            "{% if params.classification_require_fn_improvement in [false, 'False', 'false', '0', 0] %}"
            "--no-classification-require-fn-improvement "
            "{% else %}"
            "--classification-require-fn-improvement "
            "{% endif %}"
            "{% if params.classification_min_image_recall_delta not in [none, 'None', 'none', 'null', ''] %}"
            "--classification-min-image-recall-delta {{ params.classification_min_image_recall_delta }} "
            "{% endif %}"
            "{% if params.classification_min_image_ap_delta not in [none, 'None', 'none', 'null', ''] %}"
            "--classification-min-image-ap-delta {{ params.classification_min_image_ap_delta }}"
            "{% endif %}"
            "{% if params.require_mlflow_registry in [true, 'True', 'true', '1', 1] %} --require-mlflow-registry{% endif %}"
        ),
        env={
            "MLFLOW_TRACKING_URI": "{{ params.mlflow_tracking_uri }}",
            "IQA_MLFLOW_TRACKING_URI": "{{ params.mlflow_tracking_uri }}",
            "MLFLOW_S3_ENDPOINT_URL": "{{ params.mlflow_s3_endpoint_url }}",
            "IQA_S3_ENDPOINT_URL": "{{ params.s3_endpoint_url }}",
            "IQA_API_URL": "{{ params.api_url }}",
            "PYTHONPATH": "{{ params.repo_root }}:{{ params.repo_root }}/src",
        },
        pool=GPU_POOL,
        gpu_lock=True,
        repo_mount=True,
        working_dir="/opt/iqa/iqa-mlops",
        retries=0,
        execution_timeout=timedelta(hours=4),
        on_failure_callback=_clear_lifecycle_metrics_on_failure,
    )


dag = build_container_dag(
    dag_id="iqa_drift_correction_lifecycle",
    define=_define,
    schedule=None,
    tags=["iqa", "drift", "correction", "feature-ae"],
    max_active_runs=1,
    catchup=False,
    params={
        "scenario_id": "production_replay_natural_piece_b_to_piece_a_p4_drift",
        "repo_root": "/opt/iqa/iqa-mlops",
        "image_root": "/opt/iqa/iqa-mlops/.cache/iqa/source_datasets/hss-iad",
        "drift_context_path": "",
        "lifecycle_interval": 10,
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
        "api_url": "http://iqa-api:8000",
        "ml_image": ml_image(),
    },
)
