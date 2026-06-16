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
def model_version() -> dict[str, Any]:
    return {
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
        )
    )
    _record_prediction_metrics(inference_result.to_dict(), time.perf_counter() - _started)

    prediction_id = f"pred_{uuid4().hex}"
    created_at = datetime.now(timezone.utc).isoformat()
    prediction = inference_result.to_dict()
    prediction["prediction_id"] = prediction_id
    prediction["image_uri"] = request.image_uri
    prediction["model_version"] = prediction.get("feature_ae_version")
    prediction["audit_logged"] = True

    PREDICTION_STORE[prediction_id] = {
        "prediction_id": prediction_id,
        "piece_event_id": request.piece_event_id,
        "scenario_id": request.scenario_id,
        "image_uri": request.image_uri,
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


@app.post("/feedback")
def feedback(
    request: FeedbackRequest,
    x_iqa_service_token: str | None = Header(default=None, alias="X-IQA-Service-Token"),
) -> dict[str, Any]:
    _require_token("IQA_SERVICE_TOKEN", x_iqa_service_token)

    prediction = _get_open_prediction_for_feedback(request)

    if request.feedback_source == "human_sophie":
        _inc_security_metric("unsafe_train_blocked_total")
        DISPLAY_FEEDBACK_STORE[request.prediction_id] = {
            "prediction_id": request.prediction_id,
            "piece_event_id": request.piece_event_id,
            "scenario_id": request.scenario_id,
            "feedback_source": "human_sophie",
            "display_decision_source": "human_sophie",
            "train_eligibility_source": "oracle_gt",
            "eligible_for_train": False,
            "feedback_closed": False,
            "reason": "human_sophie is accepted for display only; oracle_gt remains sovereign for train eligibility.",
        }

        return {
            "accepted": True,
            "prediction_id": request.prediction_id,
            "feedback_closed": False,
            "display_decision_source": "human_sophie",
            "train_eligibility_source": "oracle_gt",
            "eligible_for_train": False,
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
    eligible_for_train = not request.gt_mask_has_defect
    if not eligible_for_train:
        _inc_security_metric("unsafe_train_blocked_total")

    FEEDBACK_STORE[request.prediction_id] = {
        "prediction_id": request.prediction_id,
        "piece_event_id": request.piece_event_id,
        "scenario_id": request.scenario_id,
        "feedback_source": "oracle_gt",
        "feedback_closed": True,
        "closed_at": closed_at,
        "verdict": verdict_dict,
        "display_decision_source": "human_sophie"
        if request.prediction_id in DISPLAY_FEEDBACK_STORE
        else "oracle_gt",
        "train_eligibility_source": "oracle_gt",
        "eligible_for_train": eligible_for_train,
    }

    display_decision_source = (
        "human_sophie"
        if request.prediction_id in DISPLAY_FEEDBACK_STORE
        else "oracle_gt"
    )

    return {
        "accepted": True,
        "prediction_id": request.prediction_id,
        "feedback_closed": True,
        "display_decision_source": display_decision_source,
        "train_eligibility_source": "oracle_gt",
        "eligible_for_train": eligible_for_train,
        "feedback": verdict_dict,
    }


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
