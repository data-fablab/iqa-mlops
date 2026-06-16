"""FastAPI gateway for IQA."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException
# from pydantic import BaseModel, Field
from iqa.api.schemas import (
    FeedbackRequest,
    PieceEventPredictRequest,
    PredictRequest,
    ReloadModelRequest,
)


from iqa.feedback import OracleFeedbackRequest, oracle_gt_verdict
from iqa.inference.contracts import InferenceRequest, placeholder_inference
from iqa.registry import ModelRegistryRef, registered_model_name
from iqa.replay import list_replay_scenarios


BASE_DIR = Path(__file__).resolve().parents[3]
ROI_MANIFEST = BASE_DIR / "models" / "manifests" / "roi_segmenter_v001_fixed" / "model_manifest.json"
FEATURE_AE_MANIFEST = BASE_DIR / "models" / "manifests" / "rd_feature_ae_gated_v001_bootstrap" / "model_manifest.json"

app = FastAPI(title="Industrial Quality Assistant API", version="0.1.0")

PREDICTION_STORE: dict[str, dict[str, Any]] = {}
FEEDBACK_STORE: dict[str, dict[str, Any]] = {}
DISPLAY_FEEDBACK_STORE: dict[str, dict[str, Any]] = {}
ADMIN_RELOAD_LOG: list[dict[str, Any]] = []

AI_SECURITY_METRICS: dict[str, int] = {
    "feedback_conflict_total": 0,
    "ai_security_incident_total": 0,
    "unsafe_train_blocked_total": 0,
    "invalid_feedback_total": 0,
    "reload_refused_total": 0,
}

# Decision/latency/ROI metrics fed by the /predict path and exposed on /metrics
# for the Grafana IQA overview dashboard (Vert/Orange/Rouge, latency, ROI fail).
PREDICTION_METRICS: dict[str, float] = {
    "decision_vert_total": 0,
    "decision_orange_total": 0,
    "decision_rouge_total": 0,
    "roi_fail_total": 0,
    "predict_latency_seconds_sum": 0.0,
    "predict_latency_seconds_count": 0,
}


# Legacy inline Pydantic schemas kept temporarily for review traceability.
# They were moved to src/iqa/api/schemas.py to centralize API contracts.
# This block can be removed after tests and review confirm the refactor.
'''
class PredictRequest(BaseModel):
    piece_event_id: str
    scenario_id: str = "production_replay_natural"
    image_uri: str = Field(..., description="S3/DVC/local URI for the primary image.")


class PieceEventPredictRequest(BaseModel):
    scenario_id: str = "production_replay_natural"
    image_uri: str = Field(..., description="S3/DVC/local URI for the primary image.")


class FeedbackRequest(BaseModel):
    piece_event_id: str
    scenario_id: str = "production_replay_natural"
    feedback_source: str = "oracle_gt"
    gt_mask_uri: str | None = None
    gt_mask_has_defect: bool = False


class ReloadModelRequest(BaseModel):
    scenario_id: str = "production_replay_natural"
    stage: str = "prod"
'''


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": "missing", "manifest_path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def _require_token(env_name: str, provided_token: str | None) -> None:
    expected = os.getenv(env_name)
    if expected and provided_token != expected:
        raise HTTPException(status_code=401, detail=f"Missing or invalid {env_name}.")


def _inc_security_metric(name: str) -> None:
    AI_SECURITY_METRICS[name] = AI_SECURITY_METRICS.get(name, 0) + 1


def _record_prediction_metrics(prediction: dict[str, Any], elapsed_seconds: float) -> None:
    decision = str(prediction.get("decision", "")).lower()
    key = f"decision_{decision}_total"
    if key in PREDICTION_METRICS:
        PREDICTION_METRICS[key] += 1
    if str(prediction.get("roi_status", "")).lower() == "fail":
        PREDICTION_METRICS["roi_fail_total"] += 1
    PREDICTION_METRICS["predict_latency_seconds_sum"] += elapsed_seconds
    PREDICTION_METRICS["predict_latency_seconds_count"] += 1


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "iqa-api"}


@app.get("/model/version")
def model_version(scenario_id: str) -> dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "registered_model_name": registered_model_name(scenario_id),
        "source_of_truth": "mlflow_registry",
        "roi_segmenter": _read_manifest(ROI_MANIFEST),
        "feature_ae": _read_manifest(FEATURE_AE_MANIFEST),
    }


@app.get("/replay-scenarios")
def replay_scenarios() -> list[dict[str, str | bool]]:
    return list_replay_scenarios()


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    _started = time.perf_counter()
    inference_result = placeholder_inference(
        InferenceRequest(
            piece_event_id=request.piece_event_id,
            scenario_id=request.scenario_id,
            image_uri=request.image_uri,
            sha256=request.sha256,
            lot_id=request.lot_id,
            source_class=request.source_class,
            dataset_version=request.dataset_version,
        )
    )
    _record_prediction_metrics(inference_result.to_dict(), time.perf_counter() - _started)

    prediction_id = f"pred_{uuid4().hex}"
    created_at = datetime.now(timezone.utc).isoformat()
    prediction = inference_result.to_dict()
    prediction["prediction_id"] = prediction_id
    prediction["image_uri"] = request.image_uri
    prediction["sha256"] = request.sha256
    prediction["lot_id"] = request.lot_id
    prediction["source_class"] = request.source_class
    prediction["dataset_version"] = request.dataset_version
    prediction["model_version"] = prediction.get("feature_ae_version")
    prediction["audit_logged"] = True

    PREDICTION_STORE[prediction_id] = {
        "prediction_id": prediction_id,
        "piece_event_id": request.piece_event_id,
        "scenario_id": request.scenario_id,
        "image_uri": request.image_uri,
        "sha256": request.sha256,
        "lot_id": request.lot_id,
        "source_class": request.source_class,
        "dataset_version": request.dataset_version,
        "decision": prediction["decision"],
        "model_version": prediction["feature_ae_version"],
        "roi_model_version": prediction["roi_model_version"],
        "created_at": created_at,
        "feedback_closed": False,
    }

    return {
        "service": "iqa-api",
        "delegated_to": "iqa-inference",
        "prediction": prediction,
        "audit": {
            "audit_logged": True,
            "prediction_id": prediction_id,
            "piece_event_id": request.piece_event_id,
            "scenario_id": request.scenario_id,
            "image_uri": request.image_uri,
            "sha256": request.sha256,
            "lot_id": request.lot_id,
            "source_class": request.source_class,
            "dataset_version": request.dataset_version,
            "decision": prediction["decision"],
            "model_version": prediction["feature_ae_version"],
            "roi_model_version": prediction["roi_model_version"],
            "created_at": created_at,
            "audit_sink": "api_response_mvp",
        },
    }


@app.post("/piece-events/{event_id}/predict")
def predict_piece_event(event_id: str, request: PieceEventPredictRequest) -> dict[str, Any]:
    return predict(
        PredictRequest(
            piece_event_id=event_id,
            scenario_id=request.scenario_id,
            image_uri=request.image_uri,
            sha256=request.sha256,
            lot_id=request.lot_id,
            source_class=request.source_class,
            dataset_version=request.dataset_version,
        )
    )


def _get_open_prediction_for_feedback(request: FeedbackRequest) -> dict[str, Any]:
    prediction = PREDICTION_STORE.get(request.prediction_id)

    if prediction is None:
        _inc_security_metric("ai_security_incident_total")
        _inc_security_metric("invalid_feedback_total")
        raise HTTPException(status_code=404, detail="Unknown prediction_id.")

    if prediction["piece_event_id"] != request.piece_event_id:
        _inc_security_metric("feedback_conflict_total")
        _inc_security_metric("ai_security_incident_total")
        raise HTTPException(status_code=409, detail="prediction_id does not match piece_event_id.")

    if prediction["scenario_id"] != request.scenario_id:
        _inc_security_metric("feedback_conflict_total")
        _inc_security_metric("ai_security_incident_total")
        raise HTTPException(status_code=409, detail="prediction_id does not match scenario_id.")

    if prediction.get("feedback_closed") is True:
        _inc_security_metric("invalid_feedback_total")
        _inc_security_metric("ai_security_incident_total")
        raise HTTPException(status_code=409, detail="Prediction already has a closed feedback.")

    return prediction


UNSAFE_TRAIN_FEEDBACK_STATUS_REASONS = {
    "defaut_confirme": "feedback_status_defaut_confirme",
    "faux_negatif": "feedback_status_faux_negatif",
    "roi_warning": "roi_warning",
    "roi_fail": "roi_fail",
}


def _feedback_status_value(feedback_status: Any) -> str | None:
    if feedback_status is None:
        return None
    return getattr(feedback_status, "value", feedback_status)


def _train_eligibility_from_feedback(request: FeedbackRequest) -> tuple[bool, str | None]:
    if request.gt_mask_has_defect:
        return False, "oracle_gt_defective"

    feedback_status = _feedback_status_value(request.feedback_status)
    if feedback_status in UNSAFE_TRAIN_FEEDBACK_STATUS_REASONS:
        return False, UNSAFE_TRAIN_FEEDBACK_STATUS_REASONS[feedback_status]

    return True, None


def _prediction_trace_context(prediction: dict[str, Any]) -> dict[str, Any]:
    return {
        "lot_id": prediction.get("lot_id"),
        "source_class": prediction.get("source_class"),
        "sha256": prediction.get("sha256"),
        "dataset_version": prediction.get("dataset_version"),
        "model_version": prediction.get("model_version"),
        "roi_model_version": prediction.get("roi_model_version"),
        "decision": prediction.get("decision"),
    }


def _prediction_audit_trail(
    *,
    prediction_id: str,
    record: dict[str, Any],
    feedback: dict[str, Any] | None,
    display_feedback: dict[str, Any] | None,
    decision: str,
    verdict: str | None,
    divergence: str | None,
) -> dict[str, Any]:
    feedback_trace = feedback or display_feedback or {}
    return {
        "prediction": {
            "prediction_id": prediction_id,
            "piece_event_id": record.get("piece_event_id"),
            "scenario_id": record.get("scenario_id"),
            "lot_id": record.get("lot_id"),
            "source_class": record.get("source_class"),
            "sha256": record.get("sha256"),
            "dataset_version": record.get("dataset_version"),
            "model_version": record.get("model_version"),
            "roi_model_version": record.get("roi_model_version"),
            "decision": decision,
        },
        "feedback": {
            "feedback_source": feedback_trace.get("feedback_source"),
            "display_feedback_source": (display_feedback or {}).get("feedback_source"),
            "display_feedback_status": (display_feedback or {}).get("feedback_status"),
            "oracle_verdict": verdict,
            "divergence": divergence,
            "train_eligibility_source": feedback_trace.get("train_eligibility_source"),
            "eligible_for_train": feedback_trace.get("eligible_for_train"),
            "train_block_reason": feedback_trace.get("train_block_reason"),
            "feedback_closed": record.get("feedback_closed", False),
            "conflict_logged": feedback_trace.get("conflict_logged", False),
        },
    }


@app.post("/feedback")
def feedback(
    request: FeedbackRequest,
    x_iqa_service_token: str | None = Header(default=None, alias="X-IQA-Service-Token"),
) -> dict[str, Any]:
    _require_token("IQA_SERVICE_TOKEN", x_iqa_service_token)

    prediction = _get_open_prediction_for_feedback(request)

    if request.feedback_source == "human_sophie":
        _inc_security_metric("unsafe_train_blocked_total")
        created_at = datetime.now(timezone.utc).isoformat()
        feedback_status = getattr(request.feedback_status, "value", request.feedback_status)
        DISPLAY_FEEDBACK_STORE[request.prediction_id] = {
            "prediction_id": request.prediction_id,
            "piece_event_id": request.piece_event_id,
            "scenario_id": request.scenario_id,
            **_prediction_trace_context(prediction),
            "feedback_source": "human_sophie",
            "feedback_status": feedback_status,
            "comment": request.comment,
            "display_decision_source": "human_sophie",
            "train_eligibility_source": "oracle_gt",
            "eligible_for_train": False,
            "train_block_reason": "human_sophie_display_only",
            "feedback_closed": False,
            "conflict_logged": False,
            "created_at": created_at,
            "reason": "human_sophie is accepted for display only; oracle_gt remains sovereign for train eligibility.",
        }

        return {
            "accepted": True,
            "prediction_id": request.prediction_id,
            "feedback_closed": False,
            "display_decision_source": "human_sophie",
            "train_eligibility_source": "oracle_gt",
            "eligible_for_train": False,
            "train_block_reason": "human_sophie_display_only",
            "conflict_logged": False,
            "created_at": created_at,
            "reason": "human_sophie is accepted for display only; oracle_gt remains sovereign for train eligibility.",
        }

    if request.feedback_source != "oracle_gt":
        _inc_security_metric("invalid_feedback_total")
        _inc_security_metric("ai_security_incident_total")
        raise HTTPException(status_code=400, detail="Unknown feedback_source.")

    verdict = oracle_gt_verdict(
        OracleFeedbackRequest(
            piece_event_id=request.piece_event_id,
            scenario_id=request.scenario_id,
            gt_mask_uri=request.gt_mask_uri,
            gt_mask_has_defect=request.gt_mask_has_defect,
        )
    )

    closed_at = datetime.now(timezone.utc).isoformat()
    prediction["feedback_closed"] = True
    prediction["feedback_closed_at"] = closed_at

    verdict_dict = verdict.to_dict()
    eligible_for_train, train_block_reason = _train_eligibility_from_feedback(request)
    if not eligible_for_train:
        _inc_security_metric("unsafe_train_blocked_total")

    display_feedback = DISPLAY_FEEDBACK_STORE.get(request.prediction_id)
    conflict_logged = display_feedback is not None
    FEEDBACK_STORE[request.prediction_id] = {
        "prediction_id": request.prediction_id,
        "piece_event_id": request.piece_event_id,
        "scenario_id": request.scenario_id,
        **_prediction_trace_context(prediction),
        "feedback_source": "oracle_gt",
        "feedback_closed": True,
        "closed_at": closed_at,
        "verdict": verdict_dict,
        "display_decision_source": "human_sophie"
        if display_feedback is not None
        else "oracle_gt",
        "train_eligibility_source": "oracle_gt",
        "eligible_for_train": eligible_for_train,
        "train_block_reason": train_block_reason,
        "conflict_logged": conflict_logged,
    }

    display_decision_source = (
        "human_sophie"
        if display_feedback is not None
        else "oracle_gt"
    )

    return {
        "accepted": True,
        "prediction_id": request.prediction_id,
        "feedback_closed": True,
        "display_decision_source": display_decision_source,
        "train_eligibility_source": "oracle_gt",
        "eligible_for_train": eligible_for_train,
        "train_block_reason": train_block_reason,
        "conflict_logged": conflict_logged,
        "feedback": verdict_dict,
    }


def _oracle_divergence(decision: str, verdict: str | None) -> str | None:
    """Classify model decision (V/O/R) against the oracle verdict.

    Returns ``None`` when no oracle feedback is closed yet. Otherwise one of:
    ``concordant``, ``faux_negatif`` (Vert mais defective = echappement),
    ``faux_positif`` (Rouge mais conforme = faux rejet), ``orange_a_revoir``.
    """

    if verdict is None:
        return None
    if decision == "Orange":
        return "orange_a_revoir"
    if decision == "Vert":
        return "faux_negatif" if verdict == "defective" else "concordant"
    if decision == "Rouge":
        return "faux_positif" if verdict == "conforme" else "concordant"
    return None


def _prediction_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prediction_id, record in PREDICTION_STORE.items():
        feedback = FEEDBACK_STORE.get(prediction_id)
        display_feedback = DISPLAY_FEEDBACK_STORE.get(prediction_id)
        feedback_trace = feedback or display_feedback or {}
        verdict = (feedback or {}).get("verdict", {}).get("verdict") if feedback else None
        decision = record.get("decision", "")
        divergence = _oracle_divergence(decision, verdict)
        audit_trail = _prediction_audit_trail(
            prediction_id=prediction_id,
            record=record,
            feedback=feedback,
            display_feedback=display_feedback,
            decision=decision,
            verdict=verdict,
            divergence=divergence,
        )
        rows.append(
            {
                "prediction_id": prediction_id,
                "piece_event_id": record.get("piece_event_id"),
                "scenario_id": record.get("scenario_id"),
                "lot_id": record.get("lot_id"),
                "source_class": record.get("source_class"),
                "sha256": record.get("sha256"),
                "dataset_version": record.get("dataset_version"),
                "decision": decision,
                "model_version": record.get("model_version"),
                "roi_model_version": record.get("roi_model_version"),
                "created_at": record.get("created_at"),
                "feedback_closed": record.get("feedback_closed", False),
                "display_decision_source": feedback_trace.get("display_decision_source"),
                "display_feedback_source": (display_feedback or {}).get("feedback_source"),
                "display_feedback_status": (display_feedback or {}).get("feedback_status"),
                "human_feedback_present": display_feedback is not None,
                "train_eligibility_source": feedback_trace.get("train_eligibility_source"),
                "eligible_for_train": feedback_trace.get("eligible_for_train"),
                "train_block_reason": feedback_trace.get("train_block_reason"),
                "conflict_logged": feedback_trace.get("conflict_logged", False),
                "oracle_verdict": verdict,
                "divergence": divergence,
                "audit_trail": audit_trail,
            }
        )
    rows.sort(key=lambda row: row.get("created_at") or "", reverse=True)
    return rows


@app.get("/predictions")
def list_predictions() -> list[dict[str, Any]]:
    """Read-only prediction history with oracle verdict and divergence flag.

    Backs Sophie's review view (lecture seule, divergence oracle).
    """

    return _prediction_rows()


@app.get("/lots/summary")
def lots_summary() -> list[dict[str, Any]]:
    """Per lot KPIs for Marc's supervision dashboard."""
    summary: dict[str, dict[str, Any]] = {}
    for row in _prediction_rows():
        lot = row.get("lot_id") or row.get("scenario_id") or "unknown"
        scenario = row.get("scenario_id") or "unknown"
        bucket = summary.setdefault(
            lot,
            {
                "lot_id": lot,
                "scenario_id": scenario,
                "total": 0,
                "vert": 0,
                "orange": 0,
                "rouge": 0,
                "feedback_closed": 0,
                "divergences": 0,
            },
        )
        if bucket.get("scenario_id") != scenario:
            bucket["scenario_id"] = "mixed"
        bucket["total"] += 1
        decision = str(row["decision"]).lower()
        if decision in {"vert", "orange", "rouge"}:
            bucket[decision] += 1
        if row["feedback_closed"]:
            bucket["feedback_closed"] += 1
        if row["divergence"] in {"faux_negatif", "faux_positif"}:
            bucket["divergences"] += 1

    rows: list[dict[str, Any]] = []
    for bucket in summary.values():
        total = bucket["total"] or 1
        bucket["taux_orange"] = round(bucket["orange"] / total, 4)
        bucket["taux_rouge"] = round(bucket["rouge"] / total, 4)
        rows.append(bucket)

    rows.sort(key=lambda row: row.get("lot_id") or row.get("scenario_id") or "")
    return rows



def _metric_label_value(value: Any) -> str:
    if value is None or value == "":
        value = "unknown"
    text = str(value)
    return text.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _metric_labels(labels: tuple[tuple[str, Any], ...]) -> str:
    return ",".join(f'{name}="{_metric_label_value(value)}"' for name, value in labels)


def _base_metric_labels(row: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    return (
        ("scenario_id", row.get("scenario_id")),
        ("lot_id", row.get("lot_id")),
        ("source_class", row.get("source_class")),
        ("model_version", row.get("model_version")),
        ("dataset_version", row.get("dataset_version")),
    )


def _count_metric(
    counters: dict[tuple[tuple[str, Any], ...], int],
    labels: tuple[tuple[str, Any], ...],
) -> None:
    counters[labels] = counters.get(labels, 0) + 1


def _append_counter_lines(
    lines: list[str],
    metric_name: str,
    counters: dict[tuple[tuple[str, Any], ...], int],
) -> None:
    for labels, value in sorted(counters.items(), key=lambda item: str(item[0])):
        lines.append(f"{metric_name}{{{_metric_labels(labels)}}} {value}")


def _filtered_metrics_lines() -> list[str]:
    prediction_counts: dict[tuple[tuple[str, Any], ...], int] = {}
    feedback_closed_counts: dict[tuple[tuple[str, Any], ...], int] = {}
    train_eligible_counts: dict[tuple[tuple[str, Any], ...], int] = {}
    divergence_counts: dict[tuple[tuple[str, Any], ...], int] = {}

    for row in _prediction_rows():
        base_labels = _base_metric_labels(row)
        _count_metric(prediction_counts, base_labels + (("decision", row.get("decision")),))

        if row.get("feedback_closed"):
            _count_metric(feedback_closed_counts, base_labels)

        if row.get("eligible_for_train") is True:
            _count_metric(train_eligible_counts, base_labels)

        divergence = row.get("divergence")
        if divergence in {"faux_negatif", "faux_positif", "orange_a_revoir"}:
            _count_metric(divergence_counts, base_labels + (("divergence", divergence),))

    lines = [
        "# HELP iqa_prediction_filtered_total IQA predictions filtered by scenario, lot, source class, model and dataset",
        "# TYPE iqa_prediction_filtered_total counter",
    ]
    _append_counter_lines(lines, "iqa_prediction_filtered_total", prediction_counts)

    lines.extend(
        [
            "# HELP iqa_feedback_closed_filtered_total IQA closed feedback filtered by scenario, lot, source class, model and dataset",
            "# TYPE iqa_feedback_closed_filtered_total counter",
        ]
    )
    _append_counter_lines(lines, "iqa_feedback_closed_filtered_total", feedback_closed_counts)

    lines.extend(
        [
            "# HELP iqa_train_eligible_filtered_total IQA train eligible feedback filtered by scenario, lot, source class, model and dataset",
            "# TYPE iqa_train_eligible_filtered_total counter",
        ]
    )
    _append_counter_lines(lines, "iqa_train_eligible_filtered_total", train_eligible_counts)

    lines.extend(
        [
            "# HELP iqa_divergence_filtered_total IQA oracle divergences filtered by scenario, lot, source class, model and dataset",
            "# TYPE iqa_divergence_filtered_total counter",
        ]
    )
    _append_counter_lines(lines, "iqa_divergence_filtered_total", divergence_counts)

    return lines

@app.get("/metrics")
def metrics() -> str:
    lines = [
        "# HELP iqa_api_up IQA API availability",
        "# TYPE iqa_api_up gauge",
        "iqa_api_up 1",
        "# HELP iqa_feedback_conflict_total IQA feedback conflicts detected by API governance",
        "# TYPE iqa_feedback_conflict_total counter",
        f"iqa_feedback_conflict_total {AI_SECURITY_METRICS['feedback_conflict_total']}",
        "# HELP iqa_ai_security_incident_total IQA AI security incidents detected by API governance",
        "# TYPE iqa_ai_security_incident_total counter",
        f"iqa_ai_security_incident_total {AI_SECURITY_METRICS['ai_security_incident_total']}",
        "# HELP iqa_unsafe_train_blocked_total IQA train eligibility blocks for unsafe or non sovereign feedback",
        "# TYPE iqa_unsafe_train_blocked_total counter",
        f"iqa_unsafe_train_blocked_total {AI_SECURITY_METRICS['unsafe_train_blocked_total']}",
        "# HELP iqa_invalid_feedback_total IQA invalid feedback events",
        "# TYPE iqa_invalid_feedback_total counter",
        f"iqa_invalid_feedback_total {AI_SECURITY_METRICS['invalid_feedback_total']}",
        "# HELP iqa_reload_refused_total IQA admin reload refusals",
        "# TYPE iqa_reload_refused_total counter",
        f"iqa_reload_refused_total {AI_SECURITY_METRICS['reload_refused_total']}",
        "# HELP iqa_prediction_total IQA predictions by decision (Vert/Orange/Rouge)",
        "# TYPE iqa_prediction_total counter",
        f'iqa_prediction_total{{decision="Vert"}} {int(PREDICTION_METRICS["decision_vert_total"])}',
        f'iqa_prediction_total{{decision="Orange"}} {int(PREDICTION_METRICS["decision_orange_total"])}',
        f'iqa_prediction_total{{decision="Rouge"}} {int(PREDICTION_METRICS["decision_rouge_total"])}',
        "# HELP iqa_roi_fail_total IQA ROI segmentation failures observed at predict time",
        "# TYPE iqa_roi_fail_total counter",
        f"iqa_roi_fail_total {int(PREDICTION_METRICS['roi_fail_total'])}",
        "# HELP iqa_predict_latency_seconds IQA predict latency (sum/count for rate-based average)",
        "# TYPE iqa_predict_latency_seconds summary",
        f"iqa_predict_latency_seconds_sum {PREDICTION_METRICS['predict_latency_seconds_sum']}",
        f"iqa_predict_latency_seconds_count {int(PREDICTION_METRICS['predict_latency_seconds_count'])}",
        "# HELP iqa_active_model_info Active IQA models served by the API (labels carry versions)",
        "# TYPE iqa_active_model_info gauge",
        (
            "iqa_active_model_info{"
            f'feature_ae_version="{_active_model_version(FEATURE_AE_MANIFEST)}",'
            f'roi_model_version="{_active_model_version(ROI_MANIFEST)}"'
            "} 1"
        ),
    ]
    lines.extend(_filtered_metrics_lines())
    return "\n".join(lines) + "\n"


def _active_model_version(manifest_path: Path) -> str:
    manifest = _read_manifest(manifest_path)
    return str(manifest.get("model_version") or manifest.get("version") or "unknown")


def _append_admin_reload_log(
    *,
    prediction_id: str | None = None,
    scenario_id: str,
    stage: Any,
    reload_status: str,
    accepted: bool,
    reason: str,
    model_name: str | None = None,
) -> dict[str, Any]:
    audit_event = {
        "reload_event_id": f"reload_{uuid4().hex}",
        "prediction_id": prediction_id,
        "scenario_id": scenario_id,
        "stage": getattr(stage, "value", stage),
        "reload_status": reload_status,
        "accepted": accepted,
        "reason": reason,
        "registered_model_name": model_name,
        "source_of_truth": "mlflow_registry",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    ADMIN_RELOAD_LOG.append(audit_event)
    return audit_event


@app.post("/admin/reload-model")
def reload_model(
    request: ReloadModelRequest,
    x_iqa_admin_token: str | None = Header(default=None, alias="X-IQA-Admin-Token"),
) -> dict[str, Any]:
    expected_token = os.getenv("IQA_ADMIN_TOKEN")

    if not expected_token:
        audit_event = _append_admin_reload_log(
            scenario_id=request.scenario_id,
            stage=request.stage,
            reload_status="refused",
            accepted=False,
            reason="IQA_ADMIN_TOKEN is not configured.",
        )
        _inc_security_metric("reload_refused_total")
        _inc_security_metric("ai_security_incident_total")
        raise HTTPException(
            status_code=503,
            detail={
                "reason": "IQA_ADMIN_TOKEN is not configured.",
                "audit_logged": True,
                "reload_event_id": audit_event["reload_event_id"],
            },
        )

    if x_iqa_admin_token != expected_token:
        audit_event = _append_admin_reload_log(
            scenario_id=request.scenario_id,
            stage=request.stage,
            reload_status="refused",
            accepted=False,
            reason="Missing or invalid IQA_ADMIN_TOKEN.",
        )
        _inc_security_metric("reload_refused_total")
        _inc_security_metric("ai_security_incident_total")
        raise HTTPException(
            status_code=401,
            detail={
                "reason": "Missing or invalid IQA_ADMIN_TOKEN.",
                "audit_logged": True,
                "reload_event_id": audit_event["reload_event_id"],
            },
        )

    model_name = registered_model_name(request.scenario_id)
    target = ModelRegistryRef(
        scenario_id=request.scenario_id,
        registered_model_name=model_name,
        stage=request.stage,
    ).to_dict()

    audit_event = _append_admin_reload_log(
        scenario_id=request.scenario_id,
        stage=request.stage,
        reload_status="accepted",
        accepted=True,
        reason="Admin reload accepted.",
        model_name=model_name,
    )

    return {
        "accepted": True,
        "reload_status": "accepted",
        "source_of_truth": "mlflow_registry",
        "audit_logged": True,
        "audit": audit_event,
        "target": target,
    }


__all__ = [
    "FeedbackRequest",
    "PieceEventPredictRequest",
    "PredictRequest",
    "ReloadModelRequest",
    "ADMIN_RELOAD_LOG",
    "AI_SECURITY_METRICS",
    "app",
    "feedback",
    "health",
    "metrics",
    "model_version",
    "predict",
    "predict_piece_event",
    "reload_model",
    "replay_scenarios",
]
