"""FastAPI gateway for IQA."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
# from pydantic import BaseModel, Field
from iqa.api.schemas import (
    Incident,
    ApiErrorResponse,
    FeedbackRequest,
    PieceEventPredictRequest,
    PredictRequest,
    ReplayRunRequest,
    ReloadModelRequest,
)


from iqa.feedback import OracleFeedbackRequest, oracle_gt_verdict
from iqa.inference.contracts import InferenceRequest, InferenceResult, placeholder_inference
from iqa.metadata.repository import MEMORY_BACKEND, MetadataRepository, create_metadata_repository, metadata_backend
from iqa.registry import ModelRegistryRef, registered_model_name
from iqa.replay import ReplayRunStore, list_replay_scenarios


BASE_DIR = Path(__file__).resolve().parents[3]
ROI_MANIFEST = BASE_DIR / "models" / "manifests" / "roi_segmenter_v001_fixed" / "model_manifest.json"
FEATURE_AE_MANIFEST = BASE_DIR / "models" / "manifests" / "rd_feature_ae_gated_v001_bootstrap" / "model_manifest.json"

app = FastAPI(title="Industrial Quality Assistant API", version="0.1.0")

PREDICTION_STORE: dict[str, dict[str, Any]] = {}
FEEDBACK_STORE: dict[str, dict[str, Any]] = {}
DISPLAY_FEEDBACK_STORE: dict[str, dict[str, Any]] = {}
ADMIN_RELOAD_LOG: list[dict[str, Any]] = []
INCIDENT_STORE: list[dict[str, Any]] = []
REPLAY_RUN_STORE = ReplayRunStore()

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
    "roi_fail_total": 0,
    "predict_latency_seconds_sum": 0.0,
    "predict_latency_seconds_count": 0,
}

# Known decision labels — bounds the cardinality of iqa_prediction_total.
PREDICTION_DECISIONS = ("Vert", "Orange", "Rouge")

# Prediction counter segregated by (scenario_id, decision) so the drift regime
# is distinguishable from natural traffic at the Prometheus level (the proxy
# drift signal filters scenario_id=~"drift.*").
PREDICTION_DECISION_COUNTS: dict[tuple[str, str], int] = {}

# PatchCore domain-drift signal (Issue 12), scored alongside the AE. The counter
# (by regime) drives the ratio alert; the gauge carries the last score for Grafana.
DOMAIN_REGIMES = ("in_domain", "out_of_domain")
DOMAIN_DRIFT_REGIME_COUNTS: dict[str, int] = {}
DOMAIN_DRIFT_METRICS: dict[str, float] = {"last_score": 0.0}

OPTIONAL_METADATA_TRACEABILITY_FIELDS = (
    "raw_dataset_id",
    "manifest_id",
    "replay_id",
    "validation_id",
    "scenario_version",
)


class MetadataWriteThrough:
    """Optional PostgreSQL journal for API metadata writes."""

    def __init__(self) -> None:
        self._backend: str | None = None
        self._repository: MetadataRepository | None = None

    def reset(self) -> None:
        self._backend = None
        self._repository = None

    def repository(self) -> MetadataRepository | None:
        backend = metadata_backend()
        if backend == MEMORY_BACKEND:
            return None
        if self._backend != backend or self._repository is None:
            self._repository = create_metadata_repository()
            self._backend = backend
        return self._repository


METADATA_WRITE_THROUGH = MetadataWriteThrough()


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


def _api_error_detail(
    *,
    status_code: int,
    error_code: str,
    message: str,
    reason: str | None = None,
    incident_type: str | None = None,
    audit_logged: bool = False,
    reload_event_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return ApiErrorResponse(
        status_code=status_code,
        error_code=error_code,
        message=message,
        reason=reason,
        incident_type=incident_type,
        audit_logged=audit_logged,
        reload_event_id=reload_event_id,
        details=details or {},
    ).model_dump(mode="json", exclude_none=True)


def _raise_api_error(
    *,
    status_code: int,
    error_code: str,
    message: str,
    reason: str | None = None,
    incident_type: str | None = None,
    audit_logged: bool = False,
    reload_event_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    raise HTTPException(
        status_code=status_code,
        detail=_api_error_detail(
            status_code=status_code,
            error_code=error_code,
            message=message,
            reason=reason,
            incident_type=incident_type,
            audit_logged=audit_logged,
            reload_event_id=reload_event_id,
            details=details,
        ),
    )


@app.exception_handler(RequestValidationError)
async def _request_validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    errors = json.loads(json.dumps(exc.errors(), default=str))
    return JSONResponse(
        status_code=422,
        content={
            "detail": _api_error_detail(
                status_code=422,
                error_code="validation_error",
                message="Request validation failed.",
                reason="Pydantic request validation failed.",
                incident_type="invalid_prediction_request",
                details={"path": str(request.url.path), "errors": errors},
            )
        },
    )


@app.exception_handler(Exception)
async def _unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "detail": _api_error_detail(
                status_code=500,
                error_code="internal_server_error",
                message="Internal server error.",
                reason="Unhandled server exception.",
                details={"path": str(request.url.path)},
            )
        },
    )


def _require_token(env_name: str, provided_token: str | None) -> None:
    expected = os.getenv(env_name)
    if expected and provided_token != expected:
        _raise_api_error(
            status_code=401,
            error_code="invalid_token",
            message=f"Missing or invalid {env_name}.",
            reason=f"Missing or invalid {env_name}.",
            incident_type="invalid_prediction_request",
        )


def _create_incident(
    *,
    incident_type: str,
    severity: str,
    message: str,
    piece_event_id: str | None = None,
    prediction_id: str | None = None,
    scenario_id: str | None = None,
    model_version: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    incident = Incident(
        incident_id=f"inc_{uuid4().hex}",
        incident_type=incident_type,
        severity=severity,
        piece_event_id=piece_event_id,
        prediction_id=prediction_id,
        scenario_id=scenario_id,
        model_version=model_version,
        message=message,
        metadata=metadata or {},
    ).model_dump(mode="json")
    INCIDENT_STORE.append(incident)
    return incident


def record_dataset_blocked_incident(
    *,
    scenario_id: str,
    dataset_version: str,
    filtered_count: int,
    sample_count: int,
    reason: str = "Candidate dataset blocked by safety filtering.",
    model_version: str | None = None,
) -> dict[str, Any]:
    return _create_incident(
        incident_type="unsafe_train_candidate_blocked",
        severity="medium",
        message=reason,
        scenario_id=scenario_id,
        model_version=model_version,
        metadata={
            "dataset_version": dataset_version,
            "filtered_count": filtered_count,
            "sample_count": sample_count,
            "reason": reason,
        },
    )


def _inc_security_metric(name: str) -> None:
    AI_SECURITY_METRICS[name] = AI_SECURITY_METRICS.get(name, 0) + 1


def _persist_metadata(
    operation: str,
    writer: Callable[[MetadataRepository], None],
    *,
    best_effort: bool = False,
) -> bool:
    try:
        repository = METADATA_WRITE_THROUGH.repository()
        if repository is None:
            return True
        writer(repository)
        return True
    except Exception as exc:
        if best_effort:
            return False
        raise HTTPException(status_code=503, detail=f"PostgreSQL metadata write failed during {operation}.") from exc


def _record_prediction_metrics(
    prediction: dict[str, Any], elapsed_seconds: float, scenario_id: str
) -> None:
    decision = str(prediction.get("decision", ""))
    if decision in PREDICTION_DECISIONS:
        counter_key = (scenario_id, decision)
        PREDICTION_DECISION_COUNTS[counter_key] = (
            PREDICTION_DECISION_COUNTS.get(counter_key, 0) + 1
        )
    if str(prediction.get("roi_status", "")).lower() == "fail":
        PREDICTION_METRICS["roi_fail_total"] += 1
    PREDICTION_METRICS["predict_latency_seconds_sum"] += elapsed_seconds
    PREDICTION_METRICS["predict_latency_seconds_count"] += 1
    regime = prediction.get("domain_regime")
    if regime in DOMAIN_REGIMES:
        DOMAIN_DRIFT_REGIME_COUNTS[regime] = DOMAIN_DRIFT_REGIME_COUNTS.get(regime, 0) + 1
    drift_score = prediction.get("domain_drift_score")
    if drift_score is not None:
        DOMAIN_DRIFT_METRICS["last_score"] = float(drift_score)


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


@app.post("/replay-runs")
def create_replay_run(request: ReplayRunRequest) -> dict[str, Any]:
    try:
        return REPLAY_RUN_STORE.create_run(request.scenario_id)
    except KeyError as exc:
        _raise_api_error(
            status_code=404,
            error_code="replay_scenario_not_found",
            message="Unknown replay scenario_id.",
            reason="Unknown replay scenario_id.",
            details={"scenario_id": request.scenario_id},
        )
        raise AssertionError("unreachable") from exc


@app.get("/replay-runs/{replay_run_id}/next")
def next_replay_event(replay_run_id: str) -> dict[str, Any]:
    try:
        return REPLAY_RUN_STORE.next_event(replay_run_id)
    except KeyError as exc:
        _raise_api_error(
            status_code=404,
            error_code="replay_run_not_found",
            message="Unknown replay_run_id.",
            reason="Unknown replay_run_id.",
            details={"replay_run_id": replay_run_id},
        )
        raise AssertionError("unreachable") from exc


@app.post("/replay-runs/{replay_run_id}/reset")
def reset_replay_run(replay_run_id: str) -> dict[str, Any]:
    try:
        return REPLAY_RUN_STORE.reset_run(replay_run_id)
    except KeyError as exc:
        _raise_api_error(
            status_code=404,
            error_code="replay_run_not_found",
            message="Unknown replay_run_id.",
            reason="Unknown replay_run_id.",
            details={"replay_run_id": replay_run_id},
        )
        raise AssertionError("unreachable") from exc


def _real_inference_enabled() -> bool:
    return os.environ.get("IQA_REAL_INFERENCE", "").strip().lower() in {"1", "true", "yes", "on"}


def _delegate_inference(request: PredictRequest) -> InferenceResult | None:
    """Delegate scoring to the GPU iqa-inference service over HTTP (stdlib only).

    Returns ``None`` on any failure so the caller falls back to the in-process
    placeholder -- the gateway never 500s the demo on an inference hiccup.
    """
    import urllib.error
    import urllib.request

    base = os.environ.get("IQA_INFERENCE_URL", "http://iqa-inference:8100").rstrip("/")
    payload = json.dumps(
        {
            "piece_event_id": request.piece_event_id,
            "scenario_id": request.scenario_id,
            "image_uri": request.image_uri,
        }
    ).encode("utf-8")
    http_request = urllib.request.Request(
        f"{base}/predict", data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(http_request, timeout=30) as response:  # noqa: S310 - internal host
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None
    return InferenceResult(
        piece_event_id=body.get("piece_event_id", request.piece_event_id),
        scenario_id=body.get("scenario_id", request.scenario_id),
        score=float(body.get("score", 0.0)),
        decision=body.get("decision", "Vert"),
        heatmap_uri=body.get("heatmap_uri"),
        roi_status=body.get("roi_status"),
        roi_model_version=body.get("roi_model_version", "roi_segmenter_v001_fixed"),
        feature_ae_version=body.get("feature_ae_version", "rd_feature_ae_gated_v001_bootstrap"),
        domain_drift_score=body.get("domain_drift_score"),
        domain_regime=body.get("domain_regime"),
    )


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    _started = time.perf_counter()
    inference_request = InferenceRequest(
        piece_event_id=request.piece_event_id,
        scenario_id=request.scenario_id,
        image_uri=request.image_uri,
        sha256=request.sha256,
        lot_id=request.lot_id,
        source_class=request.source_class,
        dataset_version=request.dataset_version,
    )
    inference_result = None
    if _real_inference_enabled():
        inference_result = _delegate_inference(request)
    if inference_result is None:
        inference_result = placeholder_inference(inference_request)
    _record_prediction_metrics(
        inference_result.to_dict(), time.perf_counter() - _started, request.scenario_id
    )

    prediction_id = f"pred_{uuid4().hex}"
    created_at = datetime.now(timezone.utc).isoformat()
    prediction = inference_result.to_dict()
    prediction["prediction_id"] = prediction_id
    prediction["image_uri"] = request.image_uri
    if request.heatmap_uri:
        prediction["heatmap_uri"] = request.heatmap_uri
    prediction["sha256"] = request.sha256
    prediction["lot_id"] = request.lot_id
    prediction["source_class"] = request.source_class
    prediction["dataset_version"] = request.dataset_version
    prediction["model_version"] = prediction.get("feature_ae_version")
    for field in OPTIONAL_METADATA_TRACEABILITY_FIELDS:
        prediction[field] = None
    prediction["audit_logged"] = True

    prediction_record = {
        "prediction_id": prediction_id,
        "piece_event_id": request.piece_event_id,
        "scenario_id": request.scenario_id,
        "image_uri": request.image_uri,
        "heatmap_uri": prediction.get("heatmap_uri"),
        "sha256": request.sha256,
        "lot_id": request.lot_id,
        "source_class": request.source_class,
        "dataset_version": request.dataset_version,
        **{field: None for field in OPTIONAL_METADATA_TRACEABILITY_FIELDS},
        "decision": prediction["decision"],
        "model_version": prediction["feature_ae_version"],
        "roi_model_version": prediction["roi_model_version"],
        "created_at": created_at,
        "feedback_closed": False,
    }
    _persist_metadata(
        "predict",
        lambda repository: repository.save_prediction(prediction_id, prediction_record),
    )
    PREDICTION_STORE[prediction_id] = prediction_record

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
            **{field: None for field in OPTIONAL_METADATA_TRACEABILITY_FIELDS},
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
            heatmap_uri=request.heatmap_uri,
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
        _raise_api_error(
            status_code=404,
            error_code="prediction_not_found",
            message="Unknown prediction_id.",
            reason="Unknown prediction_id.",
            incident_type="invalid_prediction_request",
        )

    if prediction["piece_event_id"] != request.piece_event_id:
        _inc_security_metric("feedback_conflict_total")
        _inc_security_metric("ai_security_incident_total")
        _create_incident(
            incident_type="feedback_conflict",
            severity="medium",
            message="prediction_id does not match piece_event_id.",
            piece_event_id=request.piece_event_id,
            prediction_id=request.prediction_id,
            scenario_id=request.scenario_id,
            model_version=prediction.get("model_version"),
            metadata={
                "expected_piece_event_id": prediction["piece_event_id"],
                "received_piece_event_id": request.piece_event_id,
            },
        )
        _raise_api_error(
            status_code=409,
            error_code="feedback_piece_event_mismatch",
            message="prediction_id does not match piece_event_id.",
            reason="prediction_id does not match piece_event_id.",
            incident_type="feedback_conflict",
        )

    if prediction["scenario_id"] != request.scenario_id:
        _inc_security_metric("feedback_conflict_total")
        _inc_security_metric("ai_security_incident_total")
        _create_incident(
            incident_type="feedback_conflict",
            severity="medium",
            message="prediction_id does not match scenario_id.",
            piece_event_id=request.piece_event_id,
            prediction_id=request.prediction_id,
            scenario_id=request.scenario_id,
            model_version=prediction.get("model_version"),
            metadata={
                "expected_scenario_id": prediction["scenario_id"],
                "received_scenario_id": request.scenario_id,
            },
        )
        _raise_api_error(
            status_code=409,
            error_code="feedback_scenario_mismatch",
            message="prediction_id does not match scenario_id.",
            reason="prediction_id does not match scenario_id.",
            incident_type="feedback_conflict",
        )

    if prediction.get("feedback_closed") is True:
        _inc_security_metric("invalid_feedback_total")
        _inc_security_metric("ai_security_incident_total")
        _raise_api_error(
            status_code=409,
            error_code="feedback_already_closed",
            message="Prediction already has a closed feedback.",
            reason="Prediction already has a closed feedback.",
            incident_type="invalid_prediction_request",
        )

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
            "heatmap_uri": record.get("heatmap_uri"),
            **{field: record.get(field) for field in OPTIONAL_METADATA_TRACEABILITY_FIELDS},
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
        display_feedback_record = {
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
        _persist_metadata(
            "human_sophie feedback",
            lambda repository: repository.save_display_feedback(request.prediction_id, display_feedback_record),
        )
        DISPLAY_FEEDBACK_STORE[request.prediction_id] = display_feedback_record

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
        _raise_api_error(
            status_code=400,
            error_code="unknown_feedback_source",
            message="Unknown feedback_source.",
            reason="Unknown feedback_source.",
            incident_type="invalid_prediction_request",
        )

    verdict = oracle_gt_verdict(
        OracleFeedbackRequest(
            piece_event_id=request.piece_event_id,
            scenario_id=request.scenario_id,
            gt_mask_uri=request.gt_mask_uri,
            gt_mask_has_defect=request.gt_mask_has_defect,
        )
    )

    closed_at = datetime.now(timezone.utc).isoformat()
    updated_prediction = {**prediction, "feedback_closed": True, "feedback_closed_at": closed_at}

    verdict_dict = verdict.to_dict()
    eligible_for_train, train_block_reason = _train_eligibility_from_feedback(request)
    if not eligible_for_train:
        _inc_security_metric("unsafe_train_blocked_total")

    display_feedback = DISPLAY_FEEDBACK_STORE.get(request.prediction_id)
    conflict_logged = display_feedback is not None
    feedback_record = {
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
    _persist_metadata(
        "oracle_gt feedback",
        lambda repository: repository.save_feedback_and_close_prediction(
            request.prediction_id,
            feedback_record,
            closed_at,
        ),
    )
    prediction.update(updated_prediction)
    FEEDBACK_STORE[request.prediction_id] = feedback_record

    divergence = _oracle_divergence(prediction.get("decision", ""), verdict_dict.get("verdict"))

    if divergence == "faux_negatif":
        _create_incident(
            incident_type="false_negative",
            severity="high",
            message="False negative detected by oracle_gt.",
            piece_event_id=request.piece_event_id,
            prediction_id=request.prediction_id,
            scenario_id=request.scenario_id,
            model_version=prediction.get("model_version"),
            metadata={
                "divergence": divergence,
                "decision": prediction.get("decision"),
                "oracle_verdict": verdict_dict.get("verdict"),
                "train_block_reason": train_block_reason,
            },
        )

    if train_block_reason == "roi_fail":
        _create_incident(
            incident_type="roi_fail",
            severity="high",
            message="ROI fail blocks train eligibility.",
            piece_event_id=request.piece_event_id,
            prediction_id=request.prediction_id,
            scenario_id=request.scenario_id,
            model_version=prediction.get("model_version"),
            metadata={
                "feedback_status": getattr(request.feedback_status, "value", request.feedback_status),
                "train_block_reason": train_block_reason,
            },
        )

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
                "heatmap_uri": record.get("heatmap_uri"),
                "dataset_version": record.get("dataset_version"),
                **{field: record.get(field) for field in OPTIONAL_METADATA_TRACEABILITY_FIELDS},
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


@app.get("/incidents")
def list_incidents(
    incident_type: str | None = None,
    scenario_id: str | None = None,
) -> list[dict[str, Any]]:
    rows = list(INCIDENT_STORE)
    if incident_type is not None:
        rows = [row for row in rows if row.get("incident_type") == incident_type]
    if scenario_id is not None:
        rows = [row for row in rows if row.get("scenario_id") == scenario_id]
    rows.sort(key=lambda row: row.get("created_at") or "", reverse=True)
    return rows


@app.get("/metrics", response_class=PlainTextResponse)
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
        "# HELP iqa_prediction_total IQA predictions by decision (Vert/Orange/Rouge) and scenario",
        "# TYPE iqa_prediction_total counter",
        *(
            f'iqa_prediction_total{{decision="{decision}",scenario_id="{scenario}"}} {count}'
            for (scenario, decision), count in sorted(PREDICTION_DECISION_COUNTS.items())
        ),
        "# HELP iqa_domain_drift_total IQA PatchCore domain-drift decisions by regime (in_domain/out_of_domain)",
        "# TYPE iqa_domain_drift_total counter",
        *(
            f'iqa_domain_drift_total{{regime="{regime}"}} {DOMAIN_DRIFT_REGIME_COUNTS.get(regime, 0)}'
            for regime in DOMAIN_REGIMES
        ),
        "# HELP iqa_domain_drift_score IQA PatchCore domain-drift score of the last scored piece",
        "# TYPE iqa_domain_drift_score gauge",
        f"iqa_domain_drift_score {DOMAIN_DRIFT_METRICS['last_score']}",
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
    persisted = _persist_metadata(
        "admin reload",
        lambda repository: repository.save_admin_reload_event(audit_event),
        best_effort=True,
    )
    audit_event["metadata_persisted"] = persisted
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
        _create_incident(
            incident_type="reload_refused",
            severity="high",
            message="IQA_ADMIN_TOKEN is not configured.",
            scenario_id=request.scenario_id,
            metadata={
                "reload_event_id": audit_event["reload_event_id"],
                "stage": getattr(request.stage, "value", request.stage),
                "reason": "IQA_ADMIN_TOKEN is not configured.",
            },
        )
        _raise_api_error(
            status_code=503,
            error_code="admin_token_not_configured",
            message="IQA_ADMIN_TOKEN is not configured.",
            reason="IQA_ADMIN_TOKEN is not configured.",
            incident_type="reload_refused",
            audit_logged=True,
            reload_event_id=audit_event["reload_event_id"],
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
        _create_incident(
            incident_type="reload_refused",
            severity="medium",
            message="Missing or invalid IQA_ADMIN_TOKEN.",
            scenario_id=request.scenario_id,
            metadata={
                "reload_event_id": audit_event["reload_event_id"],
                "stage": getattr(request.stage, "value", request.stage),
                "reason": "Missing or invalid IQA_ADMIN_TOKEN.",
            },
        )
        _raise_api_error(
            status_code=401,
            error_code="invalid_admin_token",
            message="Missing or invalid IQA_ADMIN_TOKEN.",
            reason="Missing or invalid IQA_ADMIN_TOKEN.",
            incident_type="reload_refused",
            audit_logged=True,
            reload_event_id=audit_event["reload_event_id"],
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
    "REPLAY_RUN_STORE",
    "record_dataset_blocked_incident",
    "create_replay_run",
    "list_incidents",
    "INCIDENT_STORE",
    "AI_SECURITY_METRICS",
    "METADATA_WRITE_THROUGH",
    "app",
    "feedback",
    "health",
    "metrics",
    "model_version",
    "predict",
    "predict_piece_event",
    "reload_model",
    "next_replay_event",
    "reset_replay_run",
    "replay_scenarios",
]
