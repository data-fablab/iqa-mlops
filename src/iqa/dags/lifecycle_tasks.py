"""IQA lifecycle DAG task implementations.

Each task is independently callable and returns context for downstream tasks.
Airflow context: context["params"] for DAG params, context["ti"].xcom_pull() for upstream outputs.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

from iqa.datasets import build_candidate_dataset, iter_manifest_image_samples
from iqa.inference.model_loader import ProdModelLoader
from iqa.monitoring import LifecycleSignal, evaluate_lifecycle_signal
from iqa.monitoring.model_metrics import (
    extract_model_quality_metrics,
    fetch_latest_quality_metrics,
    log_model_quality_metrics,
    log_per_class_quality_metrics,
    record_prod_promotion_quality,
)
from iqa.promotion import (
    evaluate_promotion_gates,
    promote_model_with_gates,
    save_previous_prod_before_promotion,
)
from iqa.registry.mlflow_registry import register_run_to_model
from iqa.roi import load_roi_mask_lookup
from iqa.training import evaluate_feature_ae_checkpoint
from iqa.training.feature_ae import FeatureAETrainingConfig
from iqa.training.feature_ae_evaluation import FeatureAEEvaluationConfig
from iqa.training.mlflow_logging import train_feature_ae_with_mlflow_logging

logger = logging.getLogger(__name__)


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _path_tuple(value: Any) -> tuple[Path, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, (str, Path)):
        return (Path(value),)
    return tuple(Path(item) for item in value)


def _load_gates_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path or "configs/promotion_gates.yaml")
    if not config_path.exists():
        return {}
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def _xcom_pull(context: dict[str, Any], task_id: str) -> Any:
    ti = context.get("ti")
    if ti is None:
        return None
    return ti.xcom_pull(task_ids=task_id)


def _skipped(reason: str, **extra: Any) -> dict[str, Any]:
    return {"status": "skipped", "reason": reason, **extra}


def _prod_quality_baseline(params: dict[str, Any]) -> dict[str, float]:
    """Read the current prod quality baseline; empty dict if unavailable.

    A missing baseline must not crash the gate: the non-regression check simply
    becomes non-evaluable (vacuous pass), which matches "no prod to regress against".
    """
    stage = params.get("prod_quality_stage", "prod")
    model_version = params.get("prod_quality_model_version")
    try:
        return fetch_latest_quality_metrics(
            stage,
            tracking_uri=params.get("mlflow_tracking_uri"),
            model_version=model_version,
        )
    except Exception as error:  # pragma: no cover - defensive (MLflow offline)
        logger.warning("Could not fetch prod quality baseline: %s", error)
        return {}


def _log_gate_verdicts(gates_result: dict[str, Any]) -> None:
    """Journal each gate verdict (and per-metric quality verdict) into the run log."""
    for name, result in gates_result.get("gates", {}).items():
        verdict = "PASS" if result.get("passed") else "FAIL"
        logger.info("gate %s: %s", name, verdict)
        if name != "quality_regression":
            continue
        quality = result.get("verdict", {})
        for metric, detail in quality.get("metrics", {}).items():
            logger.info(
                "  quality %s: %s regression=%.4f max=%.4f candidate=%.4f prod=%.4f",
                metric,
                "PASS" if detail["passed"] else "FAIL",
                detail["regression"],
                detail["max_regression"],
                detail["candidate"],
                detail["prod"],
            )
        for skipped_metric in quality.get("skipped_metrics", []):
            logger.info("  quality %s: SKIPPED (candidate or prod value absent)", skipped_metric)
        logger.info(
            "  quality decisive_metric=%s fallback_to_image_ap=%s",
            quality.get("decisive_metric"),
            quality.get("fallback_to_image_ap"),
        )


def task_lifecycle_decision(**context: Any) -> dict[str, Any]:
    """Evaluate whether this DAG run should launch a Feature-AE lifecycle."""
    params = context.get("params", {})
    signal = LifecycleSignal(
        scenario_id=params.get("scenario_id", "production_replay_natural"),
        conforming_validated_count=int(params.get("conforming_validated_count", 0)),
        drift_confirmed=bool(params.get("drift_confirmed", False)),
        roi_fail_rate=float(params.get("roi_fail_rate", 0.0)),
    )
    decision = evaluate_lifecycle_signal(signal)
    return decision.to_dict()


def task_dataset(**context: Any) -> dict[str, Any]:
    """Build candidate dataset for training.

    Reads from context["params"]:
        manifest_path, image_root, output_manifest, scenario_id, candidate_version

    Returns:
        Dict with manifest_path, dataset_version, sample_count.
    """
    params = context.get("params", {})
    decision = _xcom_pull(context, "lifecycle_decision")
    if decision is not None and not decision.get("trigger_lifecycle", False):
        return _skipped(
            "lifecycle_decision_not_triggered",
            lifecycle_decision=decision,
        )

    manifest_path = Path(params["manifest_path"])
    image_root = Path(params["image_root"])
    output_manifest = Path(params["output_manifest"])
    candidate_version = (
        params.get("candidate_version")
        or (decision or {}).get("candidate_dataset_version")
        or "v001"
    )
    manifest_version = params.get("manifest_version") or f"{candidate_version}_manifest_v001"
    roi_predictions_dirs = _path_tuple(params.get("roi_predictions_dirs"))
    roi_lookup = load_roi_mask_lookup(roi_predictions_dirs)

    samples = iter_manifest_image_samples(manifest_path, image_root)
    dataset = build_candidate_dataset(
        samples,
        output_manifest,
        version=candidate_version,
        manifest_version=manifest_version,
        roi_status=roi_lookup.status,
    )

    result = {
        "manifest_path": str(dataset.output_manifest),
        "dataset_version": dataset.version,
        "manifest_version": manifest_version,
        "sample_count": dataset.sample_count,
        "filtered_count": dataset.filtered_count,
        "roi_status_count": len(roi_lookup.status),
        "lifecycle_decision": decision,
    }
    if not roi_predictions_dirs:
        result["warning"] = "roi_predictions_dirs not provided; ROI status filtering was not applied"
    return result


def task_train(**context: Any) -> dict[str, Any]:
    """Train Feature AE model on candidate dataset.

    Reads from context["ti"].xcom_pull(task_ids="dataset"):
        manifest_path, dataset_version
    Reads from context["params"]:
        scenario_id, image_root, output_checkpoint, candidate_version

    Returns:
        Dict with run_id, checkpoint, run_dir.
    """
    params = context.get("params", {})
    ti = context["ti"]
    dataset = ti.xcom_pull(task_ids="dataset")
    if dataset.get("status") == "skipped":
        return _skipped(dataset["reason"], lifecycle_decision=dataset.get("lifecycle_decision"))

    config = FeatureAETrainingConfig(
        manifest_path=Path(dataset["manifest_path"]),
        image_root=Path(params["image_root"]),
        output_checkpoint=Path(params["output_checkpoint"]),
        scenario_id=params.get("scenario_id", "production_replay_natural"),
        dataset_version=dataset.get("dataset_version", ""),
        manifest_version=dataset.get("manifest_version", ""),
        candidate_version=params.get("candidate_version", dataset.get("dataset_version", "")),
        roi_model_version=params.get("roi_model_version", "roi_segmenter_v001_fixed"),
        feature_ae_version=params.get("feature_ae_version", "rd_feature_ae_gated_v001_bootstrap"),
    )

    result = train_feature_ae_with_mlflow_logging(config, git_commit=_get_git_commit())

    return {
        "run_id": result.get("run_id", ""),
        "checkpoint": result.get("checkpoint", str(config.output_checkpoint)),
        "run_dir": result.get("run_dir", ""),
        "dataset_version": config.dataset_version,
        "manifest_version": config.manifest_version,
    }


def task_eval(**context: Any) -> dict[str, Any]:
    """Evaluate Feature AE on validation set.

    Reads from context["ti"].xcom_pull(task_ids="train"):
        checkpoint
    Reads from context["params"]:
        manifest_path, image_root, output_dir, validation_set_id

    Returns:
        Dict with metrics: image_recall, image_ap, orange_rate, latency_ms, plus
        the model-quality run id where the 4 business metrics were logged.
    """
    params = context.get("params", {})
    ti = context["ti"]
    train_output = ti.xcom_pull(task_ids="train")
    if train_output.get("status") == "skipped":
        return _skipped(train_output["reason"], lifecycle_decision=train_output.get("lifecycle_decision"))

    checkpoint = Path(train_output["checkpoint"])
    output_dir = Path(params.get("eval_output_dir", str(checkpoint.parent / "eval")))

    config = FeatureAEEvaluationConfig(
        checkpoint_path=checkpoint,
        manifest_path=Path(params["manifest_path"]),
        image_root=Path(params["image_root"]),
        output_dir=output_dir,
        validation_set_id=params.get("validation_set_id", "validation_set_v001"),
    )

    result = evaluate_feature_ae_checkpoint(config)
    metrics = result["metrics"]

    # Log the canonical business metrics (AUPIMO, pixel_ap, image_ap, image_auroc)
    # to the iqa-model-quality experiment so the Grafana dashboards (Issues 1-3) and
    # the candidate-vs-prod gate (Issue 4) have an up-to-date source for this run.
    model_version = (
        params.get("candidate_version")
        or train_output.get("dataset_version")
        or params.get("feature_ae_version", "")
    )
    stage = params.get("eval_stage", "candidate")
    tracking_uri = params.get("mlflow_tracking_uri")
    model_quality_run_id = log_model_quality_metrics(
        metrics,
        model_version=model_version,
        stage=stage,
        tracking_uri=tracking_uri,
    )

    # Per-class metrics (class1/class2/class3) make incremental drift coverage
    # visible in Grafana; attach them to the same model-quality run (Issue 4).
    per_class_metrics = result.get("per_class_metrics", {})
    if per_class_metrics:
        log_per_class_quality_metrics(
            per_class_metrics,
            run_id=model_quality_run_id,
            tracking_uri=tracking_uri,
        )

    # The 4 business metrics drive the candidate-vs-prod non-regression gate; carry
    # them forward so task_gates can compare against the prod baseline.
    quality_metrics = extract_model_quality_metrics(metrics)

    return {
        "recall": metrics.get("image_recall", 0.0),
        "ap": metrics.get("image_ap") or 0.0,
        "orange_rate": metrics.get("orange_rate", 0.0),
        "latency_ms": metrics.get("latency_ms", 0.0),
        "false_negatives": metrics.get("false_negatives", 0),
        "model_quality_run_id": model_quality_run_id,
        "model_version": model_version,
        "stage": stage,
        "quality_metrics": quality_metrics,
        "per_class_metrics": per_class_metrics,
    }


def task_gates(**context: Any) -> dict[str, Any]:
    """Check promotion gates on candidate metrics.

    Reads from context["ti"].xcom_pull(task_ids="eval"):
        recall, ap, orange_rate, latency_ms

    Raises:
        Exception: If any gate fails (blocking).

    Returns:
        Dict with gate evaluation results.
    """
    params = context.get("params", {})
    ti = context["ti"]
    eval_output = ti.xcom_pull(task_ids="eval")
    if eval_output.get("status") == "skipped":
        return _skipped(eval_output["reason"], lifecycle_decision=eval_output.get("lifecycle_decision"))
    gates_config = _load_gates_config(params.get("gates_config_path"))

    # When the candidate's 4 business metrics are available, gate on non-regression
    # vs the prod baseline (ADR 0010 §6) instead of the legacy single-AP regression.
    candidate_quality = eval_output.get("quality_metrics") or None
    prod_quality = _prod_quality_baseline(params) if candidate_quality else None

    gates_result = evaluate_promotion_gates(
        candidate_recall=eval_output.get("recall", 0.0),
        candidate_ap=eval_output.get("ap", 0.0),
        candidate_orange_rate=eval_output.get("orange_rate", 0.0),
        candidate_latency_ms=eval_output.get("latency_ms", 0.0),
        prod_ap=eval_output.get("prod_ap"),
        gates_config=gates_config,
        candidate_quality_metrics=candidate_quality,
        prod_quality_metrics=prod_quality,
    )

    _log_gate_verdicts(gates_result)

    if not gates_result["all_passed"]:
        gates_result["status"] = "rejected"
        gates_result["rejection_reason"] = _summarize_gate_failures(gates_result)
        logger.warning(
            "candidate rejected: %s", gates_result["rejection_reason"]
        )
        raise Exception(
            f"Gates failed: {gates_result['rejection_reason']}. Candidate logged as rejected. Promotion blocked."
        )

    return gates_result


def _summarize_gate_failures(gates_result: dict[str, Any]) -> str:
    """One-line summary of which gates failed (for logging / anti-loop)."""
    failed = [
        name
        for name, result in gates_result.get("gates", {}).items()
        if not result.get("passed")
    ]
    return f"gates_failed={','.join(failed)}" if failed else "unknown_gate_failure"


def task_mlflow(**context: Any) -> dict[str, Any]:
    """Register trained model in MLflow as candidate.

    Reads from context["ti"].xcom_pull(task_ids="train"):
        run_id
    Reads from context["params"]:
        scenario_id

    Returns:
        Dict with registered_model_name, version, stage.
    """
    params = context.get("params", {})
    ti = context["ti"]
    train_output = ti.xcom_pull(task_ids="train")
    if train_output.get("status") == "skipped":
        return _skipped(train_output["reason"], lifecycle_decision=train_output.get("lifecycle_decision"))

    run_id = train_output.get("run_id")
    if not run_id:
        raise ValueError("run_id required (from task_train via XCom)")

    scenario_id = params.get("scenario_id", "production_replay_natural")

    result = register_run_to_model(
        run_id=run_id,
        scenario_id=scenario_id,
        stage="candidate",
    )
    result["dataset_version"] = train_output.get("dataset_version", "")
    result["manifest_version"] = train_output.get("manifest_version", "")
    return result


def task_promotion(**context: Any) -> dict[str, Any]:
    """Promote candidate model to test stage.

    Reads from context["ti"].xcom_pull(task_ids="mlflow"):
        registered_model_name, version
    Reads from context["ti"].xcom_pull(task_ids="eval"):
        recall, ap, orange_rate, latency_ms

    Raises:
        Exception: If gates fail during promotion.

    Returns:
        Dict with promotion result (success, transition, artifacts).
    """
    params = context.get("params", {})
    ti = context["ti"]
    mlflow_output = ti.xcom_pull(task_ids="mlflow")
    eval_output = ti.xcom_pull(task_ids="eval")
    if mlflow_output.get("status") == "skipped":
        return _skipped(mlflow_output["reason"], lifecycle_decision=mlflow_output.get("lifecycle_decision"))

    registered_model_name_str = mlflow_output.get("registered_model_name")
    version = mlflow_output.get("version")

    if not registered_model_name_str or not version:
        raise ValueError("registered_model_name and version required (from task_mlflow via XCom)")

    candidate_metrics = {
        "recall": eval_output.get("recall", 0.0),
        "ap": eval_output.get("ap", 0.0),
        "orange_rate": eval_output.get("orange_rate", 0.0),
        "latency_ms": eval_output.get("latency_ms", 0.0),
    }
    target_stage = params.get("target_stage", "test")
    if target_stage == "prod":
        previous_prod = save_previous_prod_before_promotion(registered_model_name_str)
        if not previous_prod["success"]:
            raise Exception(f"Could not save previous prod before promotion: {previous_prod.get('error')}")

    # Re-evaluate the same 4-metric non-regression gate at promotion time so the
    # MLflow transition only happens when the candidate does not regress vs prod.
    candidate_quality = eval_output.get("quality_metrics") or None
    prod_quality = _prod_quality_baseline(params) if candidate_quality else None

    result = promote_model_with_gates(
        registered_model_name=registered_model_name_str,
        version=version,
        target_stage=target_stage,
        candidate_metrics=candidate_metrics,
        prod_metrics={"ap": eval_output["prod_ap"]} if "prod_ap" in eval_output else None,
        gates_config_path=params.get("gates_config_path"),
        candidate_quality_metrics=candidate_quality,
        prod_quality_metrics=prod_quality,
    )

    if not result["success"]:
        raise Exception(
            f"Promotion blocked: {result.get('blocked_reasons', 'unknown reason')}"
        )

    # On a successful prod promotion, advance the quality baseline so the exporter
    # exposes prod vs previous_prod gauges for the regression rule (Issue 5).
    if target_stage == "prod" and candidate_quality:
        try:
            result["quality_baseline"] = record_prod_promotion_quality(
                candidate_quality,
                model_version=eval_output.get("model_version", ""),
                tracking_uri=params.get("mlflow_tracking_uri"),
            )
        except Exception as error:  # pragma: no cover - defensive (MLflow offline)
            logger.warning("Could not record prod quality baseline: %s", error)

    return result


def _notify_inference_service_reload() -> dict[str, Any]:
    """POST /reload-model on the inference service so it drops its cached scorer.

    Best-effort: if the service is unreachable (e.g. not running during a test),
    the reload still succeeds from MLflow's perspective. The next ``/predict``
    will load the promoted checkpoint lazily.
    """
    import json as _json
    import urllib.error
    import urllib.request

    base = os.environ.get("IQA_INFERENCE_URL", "http://iqa-inference:8100").rstrip("/")
    try:
        req = urllib.request.Request(
            f"{base}/reload-model",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310 - internal host
            return _json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        logger.warning("inference service /reload-model unreachable: %s", exc)
        return {"status": "unreachable", "error": str(exc)}


def task_reload(**context: Any) -> dict[str, Any]:
    """Reload production model in inference service after promotion.

    Calls ``ProdModelLoader.reload()`` to fetch the promoted model from MLflow,
    then notifies the running inference service via ``POST /reload-model`` so it
    drops its cached scorer and picks up the new checkpoint (Issue 10).
    """
    params = context.get("params", {})
    target_stage = params.get("target_stage", "test")
    if target_stage != "prod":
        return {
            "status": "skipped",
            "reason": f"target_stage is {target_stage}; production reload only runs for prod promotions",
            "target_stage": target_stage,
        }

    scenario_id = params.get("scenario_id", "production_replay_natural")

    loader = ProdModelLoader(scenario_id)
    loaded_model = loader.reload()

    inference_reload = _notify_inference_service_reload()

    return {
        "version": loaded_model.version,
        "artifact_uri": loaded_model.artifact_uri,
        "registered_model_name": loaded_model.registered_model_name,
        "inference_service_reload": inference_reload,
    }


__all__ = [
    "task_lifecycle_decision",
    "task_dataset",
    "task_train",
    "task_eval",
    "task_gates",
    "task_mlflow",
    "task_promotion",
    "task_reload",
]
