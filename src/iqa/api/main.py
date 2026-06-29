"""FastAPI gateway for IQA."""

from __future__ import annotations

import json
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable
from typing import Any
from uuid import uuid4
from urllib.error import HTTPError, URLError
from urllib.request import Request as UrlRequest, urlopen

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
# from pydantic import BaseModel, Field
from iqa.api.schemas import (
    Incident,
    ApiErrorResponse,
    DriftEventRequest,
    FeedbackRequest,
    LifecycleEventRequest,
    PieceEventPredictRequest,
    PredictRequest,
    ReplayRunRequest,
    ReloadModelRequest,
)


from iqa.feedback import OracleFeedbackRequest, oracle_gt_verdict
from iqa.inference.contracts import InferenceRequest, InferenceResult
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
    "decision_vert_total": 0,
    "decision_orange_total": 0,
    "decision_rouge_total": 0,
    "roi_fail_total": 0,
    "predict_latency_seconds_sum": 0.0,
    "predict_latency_seconds_count": 0,
}

LIFECYCLE_STATE: dict[str, Any] = {
    "current": {},
    "epoch_metrics": {},
    "epoch_updated_at": 0.0,
    "gate_metrics": {},
    "gate_values": {},
    "gate_deltas": {},
    "active_models": {},
    "final_models": {},
    "summary_metrics": {},
    "promotion_decisions": {},
    "promotion_seen": set(),
    "promotion_counters": {},
}
LIFECYCLE_EPOCH_METRIC_ALIASES = {
    "pixel_aupimo_1e-5_1e-3": "pixel_aupimo",
    "pixel_aupimo": "pixel_aupimo",
    "pixel_ap": "pixel_ap",
    "image_ap": "image_ap",
    "false_negatives": "false_negatives",
}
LIFECYCLE_GATE_VALUE_METRICS = {
    "pixel_aupimo",
    "pixel_ap",
    "image_ap",
    "image_recall",
    "false_negatives",
}
LIFECYCLE_ALLOWED_METRICS = {
    "pixel_aupimo_1e-5_1e-3",
    "pixel_aupimo",
    "pixel_ap",
    "image_ap",
    "false_negatives",
    "events_processed",
    "cycles_completed",
    "localization_metric_delta",
    "classification_metric_delta",
    "classification_fn_delta",
    "gate_metric_delta",
    "gate_fn_delta",
}
LIFECYCLE_SENSITIVE_KEYS = (
    "image",
    "path",
    "uri",
    "mask",
    "heatmap",
    "piece_event",
    "relative",
)

DRIFT_STATE: dict[str, Any] = {
    "current": {},
}
DRIFT_ACTIVE_MODEL_ALLOWED_FIELDS = {
    "version",
    "registry_model_name",
    "registered_model_version",
    "registry_stage",
    "runtime_contract_status",
}
DRIFT_ALLOWED_METRICS = {
    "drift_score",
    "window_events",
    "window_index",
    "first_confirmed_window_index",
    "alert_rate",
    "red_rate",
    "unexpected_red_rate",
    "roi_fail_rate",
    "oracle_fn_rate",
    "domain_ratio",
    "domain_score",
    "degradation_score",
}
DRIFT_ALLOWED_STATUSES = {"clear", "suspected", "confirmed"}
DRIFT_ALLOWED_MODEL_ROLES = {"classification", "localization"}
OBSERVABILITY_TRANSIENT_TTL_SECONDS = float(os.environ.get("IQA_OBSERVABILITY_TRANSIENT_TTL_SECONDS", "30"))

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


def _inference_http_error_detail(error: HTTPError) -> str:
    try:
        payload = json.loads(error.read().decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return str(error.reason or f"HTTP {error.code}")

    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, str):
            return detail
        if detail is not None:
            return json.dumps(detail, ensure_ascii=False)

    return str(error.reason or f"HTTP {error.code}")


def _call_inference_service(request: InferenceRequest) -> InferenceResult:
    base_url = os.environ.get(
        "IQA_INFERENCE_URL",
        "http://iqa-inference:8100",
    ).strip().rstrip("/")

    if not base_url:
        _raise_api_error(
            status_code=503,
            error_code="inference_service_configuration_error",
            message="Inference service URL is not configured.",
            reason="IQA_INFERENCE_URL is empty.",
            details={"path": "/predict"},
        )

    timeout_raw = os.environ.get("IQA_INFERENCE_TIMEOUT_SECONDS", "120")
    try:
        timeout = float(timeout_raw)
    except ValueError:
        _raise_api_error(
            status_code=503,
            error_code="inference_service_configuration_error",
            message="Inference service timeout is invalid.",
            reason=f"Invalid IQA_INFERENCE_TIMEOUT_SECONDS: {timeout_raw!r}.",
            details={"path": "/predict"},
        )
        raise AssertionError("unreachable")

    if timeout <= 0:
        _raise_api_error(
            status_code=503,
            error_code="inference_service_configuration_error",
            message="Inference service timeout is invalid.",
            reason="IQA_INFERENCE_TIMEOUT_SECONDS must be greater than zero.",
            details={"path": "/predict"},
        )

    payload = {
        "piece_event_id": request.piece_event_id,
        "scenario_id": request.scenario_id,
        "image_uri": request.image_uri,
        "sha256": request.sha256,
        "lot_id": request.lot_id,
        "source_class": request.source_class,
        "dataset_version": request.dataset_version,
    }
    http_request = UrlRequest(
        f"{base_url}/predict",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(http_request, timeout=timeout) as response:
            raw_response = response.read()
    except HTTPError as error:
        detail = _inference_http_error_detail(error)

        if error.code == 404:
            _raise_api_error(
                status_code=404,
                error_code="inference_input_not_found",
                message="Inference input image was not found.",
                reason=detail,
                details={"path": "/predict", "upstream_status": error.code},
            )
        if error.code == 422:
            _raise_api_error(
                status_code=422,
                error_code="inference_input_invalid",
                message="Inference input was rejected.",
                reason=detail,
                details={"path": "/predict", "upstream_status": error.code},
            )
        if error.code == 503:
            _raise_api_error(
                status_code=503,
                error_code="inference_service_unavailable",
                message="Inference service is unavailable.",
                reason=detail,
                details={"path": "/predict", "upstream_status": error.code},
            )

        _raise_api_error(
            status_code=502,
            error_code="invalid_inference_response",
            message="Inference service returned an unexpected HTTP response.",
            reason=detail,
            details={"path": "/predict", "upstream_status": error.code},
        )
    except socket.timeout as error:
        _raise_api_error(
            status_code=504,
            error_code="inference_service_timeout",
            message="Inference service timed out.",
            reason=str(error),
            details={"path": "/predict"},
        )
    except URLError as error:
        if isinstance(error.reason, (socket.timeout, TimeoutError)):
            _raise_api_error(
                status_code=504,
                error_code="inference_service_timeout",
                message="Inference service timed out.",
                reason=str(error.reason),
                details={"path": "/predict"},
            )

        _raise_api_error(
            status_code=503,
            error_code="inference_service_unavailable",
            message="Inference service is unavailable.",
            reason=str(error.reason),
            details={"path": "/predict"},
        )

    try:
        response_payload = json.loads(raw_response.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        _raise_api_error(
            status_code=502,
            error_code="invalid_inference_response",
            message="Inference service returned invalid JSON.",
            reason=str(error),
            details={"path": "/predict"},
        )
        raise AssertionError("unreachable")

    if not isinstance(response_payload, dict):
        _raise_api_error(
            status_code=502,
            error_code="invalid_inference_response",
            message="Inference service returned an invalid contract.",
            reason="Expected a JSON object.",
            details={"path": "/predict"},
        )

    required_fields = {
        "piece_event_id",
        "scenario_id",
        "score",
        "decision",
        "heatmap_uri",
        "roi_status",
        "roi_model_version",
        "feature_ae_version",
    }
    missing = sorted(required_fields - set(response_payload))
    if missing:
        _raise_api_error(
            status_code=502,
            error_code="invalid_inference_response",
            message="Inference service returned an incomplete contract.",
            reason=f"Missing fields: {', '.join(missing)}.",
            details={"path": "/predict"},
        )

    decision = response_payload["decision"]
    if decision not in {"Vert", "Orange", "Rouge"}:
        _raise_api_error(
            status_code=502,
            error_code="invalid_inference_response",
            message="Inference service returned an invalid decision.",
            reason=f"Unsupported decision: {decision!r}.",
            details={"path": "/predict"},
        )

    if (
        response_payload["piece_event_id"] != request.piece_event_id
        or response_payload["scenario_id"] != request.scenario_id
    ):
        _raise_api_error(
            status_code=502,
            error_code="invalid_inference_response",
            message="Inference service returned mismatched traceability identifiers.",
            reason="piece_event_id or scenario_id does not match the request.",
            details={"path": "/predict"},
        )

    try:
        return InferenceResult(
            piece_event_id=str(response_payload["piece_event_id"]),
            scenario_id=str(response_payload["scenario_id"]),
            score=float(response_payload["score"]),
            decision=decision,
            heatmap_uri=(
                None
                if response_payload["heatmap_uri"] is None
                else str(response_payload["heatmap_uri"])
            ),
            roi_status=(
                None
                if response_payload["roi_status"] is None
                else str(response_payload["roi_status"])
            ),
            roi_model_version=str(response_payload["roi_model_version"]),
            feature_ae_version=str(response_payload["feature_ae_version"]),
        )
    except (TypeError, ValueError) as error:
        _raise_api_error(
            status_code=502,
            error_code="invalid_inference_response",
            message="Inference service returned an invalid contract.",
            reason=str(error),
            details={"path": "/predict"},
        )
        raise AssertionError("unreachable")


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



def _read_metadata(
    operation: str,
    reader: Callable[[MetadataRepository], Any],
    *,
    default: Any = None,
) -> Any:
    try:
        repository = METADATA_WRITE_THROUGH.repository()
        if repository is None:
            return default
        return reader(repository)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"PostgreSQL metadata read failed during {operation}.",
        ) from exc


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


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    _started = time.perf_counter()
    inference_result = _call_inference_service(
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
    prediction = _read_metadata(
        "prediction lookup",
        lambda repository: repository.get_prediction(request.prediction_id),
    )
    if prediction is None:
        prediction = PREDICTION_STORE.get(request.prediction_id)
    else:
        PREDICTION_STORE[request.prediction_id] = prediction

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

    display_feedback = _read_metadata(
        "display feedback lookup",
        lambda repository: repository.get_display_feedback(request.prediction_id),
    )
    if display_feedback is None:
        display_feedback = DISPLAY_FEEDBACK_STORE.get(request.prediction_id)
    else:
        DISPLAY_FEEDBACK_STORE[request.prediction_id] = display_feedback

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
    persisted_predictions = _read_metadata(
        "prediction history lookup",
        lambda repository: repository.list_predictions(),
        default=[],
    )

    records: dict[str, dict[str, Any]] = {}

    for record in persisted_predictions:
        prediction_id = record.get("prediction_id")
        if prediction_id:
            records[prediction_id] = record
            PREDICTION_STORE[prediction_id] = record

    for prediction_id, record in PREDICTION_STORE.items():
        records.setdefault(prediction_id, record)

    rows: list[dict[str, Any]] = []

    for prediction_id, record in records.items():
        feedback = _read_metadata(
            "oracle feedback lookup",
            lambda repository, prediction_id=prediction_id: repository.get_feedback(
                prediction_id
            ),
        )
        if feedback is None:
            feedback = FEEDBACK_STORE.get(prediction_id)
        else:
            FEEDBACK_STORE[prediction_id] = feedback

        display_feedback = _read_metadata(
            "display feedback lookup",
            lambda repository, prediction_id=prediction_id: repository.get_display_feedback(
                prediction_id
            ),
        )
        if display_feedback is None:
            display_feedback = DISPLAY_FEEDBACK_STORE.get(prediction_id)
        else:
            DISPLAY_FEEDBACK_STORE[prediction_id] = display_feedback

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
                **{
                    field: record.get(field)
                    for field in OPTIONAL_METADATA_TRACEABILITY_FIELDS
                },
                "decision": decision,
                "model_version": record.get("model_version"),
                "roi_model_version": record.get("roi_model_version"),
                "created_at": record.get("created_at"),
                "feedback_closed": record.get("feedback_closed", False),
                "display_decision_source": feedback_trace.get(
                    "display_decision_source"
                ),
                "display_feedback_source": (display_feedback or {}).get(
                    "feedback_source"
                ),
                "display_feedback_status": (display_feedback or {}).get(
                    "feedback_status"
                ),
                "human_feedback_present": display_feedback is not None,
                "train_eligibility_source": feedback_trace.get(
                    "train_eligibility_source"
                ),
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


def _cycle_number(cycle_id: Any) -> int:
    text = str(cycle_id or "")
    if text.startswith("cycle_"):
        text = text.rsplit("_", 1)[-1]
    try:
        return int(text)
    except ValueError:
        return 0


def _finite_metric(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def _observability_is_recent(updated_at: Any) -> bool:
    if OBSERVABILITY_TRANSIENT_TTL_SECONDS <= 0:
        return True
    try:
        timestamp = float(updated_at)
    except (TypeError, ValueError):
        return False
    return time.time() - timestamp <= OBSERVABILITY_TRANSIENT_TTL_SECONDS


def _lifecycle_base_labels(current: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    return (
        ("scenario_id", current.get("scenario_id")),
        ("lifecycle_run_id", current.get("lifecycle_run_id")),
        ("cycle_id", current.get("cycle_id")),
        ("candidate_version", current.get("candidate_version")),
        ("candidate_init_policy", current.get("candidate_init_policy")),
    )


def _is_allowed_lifecycle_metric_name(name: str) -> bool:
    if name in LIFECYCLE_ALLOWED_METRICS:
        return True
    if _parse_gate_value_metric_name(name) is not None:
        return True
    if _parse_gate_delta_metric_name(name) is not None:
        return True
    return False


def _parse_gate_value_metric_name(name: str) -> tuple[str, str, str] | None:
    prefix = "gate_"
    if not name.startswith(prefix):
        return None
    remainder = name[len(prefix) :]
    for role in ("localization", "classification"):
        role_prefix = f"{role}_"
        if not remainder.startswith(role_prefix):
            continue
        model_and_metric = remainder[len(role_prefix) :]
        for model in ("active", "candidate"):
            model_prefix = f"{model}_"
            if not model_and_metric.startswith(model_prefix):
                continue
            metric = model_and_metric[len(model_prefix) :]
            if metric in LIFECYCLE_GATE_VALUE_METRICS:
                return role, model, metric
    return None


def _parse_gate_delta_metric_name(name: str) -> tuple[str, str] | None:
    prefix = "gate_delta_"
    if not name.startswith(prefix):
        return None
    remainder = name[len(prefix) :]
    for role in ("localization", "classification"):
        role_prefix = f"{role}_"
        if not remainder.startswith(role_prefix):
            continue
        metric = remainder[len(role_prefix) :]
        if metric in LIFECYCLE_GATE_VALUE_METRICS:
            return role, metric
    return None


def _reject_sensitive_lifecycle_payload(payload: dict[str, Any]) -> None:
    def visit(value: Any, *, key: str = "") -> None:
        lowered_key = key.lower()
        if any(marker in lowered_key for marker in LIFECYCLE_SENSITIVE_KEYS):
            _raise_api_error(
                status_code=422,
                error_code="sensitive_lifecycle_field",
                message="Lifecycle event contains a sensitive or non-observable field.",
                reason=f"Lifecycle field {key!r} is not accepted by the metrics API.",
            )
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                if key == "metrics" and not _is_allowed_lifecycle_metric_name(str(child_key)):
                    _raise_api_error(
                        status_code=422,
                        error_code="unsupported_lifecycle_metric",
                        message="Lifecycle event contains an unsupported metric.",
                        reason=f"Metric {child_key!r} is not accepted by the metrics API.",
                    )
                visit(child_value, key="" if key == "metrics" else str(child_key))
            return
        if isinstance(value, list):
            for item in value:
                visit(item, key=key)
            return
        if isinstance(value, str):
            text = value.lower()
            if (
                "://" in text
                or ":\\" in text
                or "\\\\" in text
                or "/.cache/" in text
                or "\\.cache\\" in text
                or "/data/" in text
                or "\\data\\" in text
                or "/models/" in text
                or "\\models\\" in text
            ):
                _raise_api_error(
                    status_code=422,
                    error_code="sensitive_lifecycle_value",
                    message="Lifecycle event contains a sensitive value.",
                    reason=f"Lifecycle value for {key!r} looks like a path or URI.",
                )

    visit(payload)


def _record_lifecycle_observation(event: LifecycleEventRequest) -> None:
    payload = event.model_dump(mode="json", exclude_none=True)
    _reject_sensitive_lifecycle_payload(payload)
    current = LIFECYCLE_STATE["current"]
    now = time.time()
    current["_updated_at"] = now
    for key in (
        "scenario_id",
        "lifecycle_run_id",
        "cycle_id",
        "candidate_version",
        "candidate_init_policy",
    ):
        if payload.get(key) is not None:
            current[key] = payload[key]
    if event.epoch is not None:
        current["epoch"] = event.epoch
    metrics_payload = {
        key: value
        for key, value in (payload.get("metrics") or {}).items()
        if _is_allowed_lifecycle_metric_name(str(key)) and _finite_metric(value) is not None
    }
    if event.event_type == "epoch_completed":
        epoch_metrics = LIFECYCLE_STATE["epoch_metrics"]
        epoch_metrics.clear()
        LIFECYCLE_STATE["epoch_updated_at"] = now
        for source_name, value in metrics_payload.items():
            normalized_name = LIFECYCLE_EPOCH_METRIC_ALIASES.get(source_name)
            if normalized_name is not None:
                epoch_metrics[normalized_name] = value
    if event.event_type in {"gate_decision", "promotion_decision"}:
        gate_metrics = LIFECYCLE_STATE["gate_metrics"]
        if "localization_metric_delta" in metrics_payload:
            gate_metrics.setdefault("localization", {})["metric_delta"] = metrics_payload["localization_metric_delta"]
        if "classification_metric_delta" in metrics_payload:
            gate_metrics.setdefault("classification", {})["metric_delta"] = metrics_payload["classification_metric_delta"]
        if "classification_fn_delta" in metrics_payload:
            gate_metrics.setdefault("classification", {})["fn_delta"] = metrics_payload["classification_fn_delta"]
        for name, value in metrics_payload.items():
            parsed_gate_value = _parse_gate_value_metric_name(str(name))
            if parsed_gate_value is not None:
                role, model, metric_name = parsed_gate_value
                labels = _lifecycle_gate_labels(event, role=role, model=model, metric_name=metric_name)
                LIFECYCLE_STATE["gate_values"][labels] = value
                continue
            parsed_gate_delta = _parse_gate_delta_metric_name(str(name))
            if parsed_gate_delta is not None:
                role, metric_name = parsed_gate_delta
                labels = _lifecycle_gate_delta_labels(event, role=role, metric_name=metric_name)
                LIFECYCLE_STATE["gate_deltas"][labels] = value
        _record_lifecycle_promotion_status(event)
    summary_metrics = LIFECYCLE_STATE["summary_metrics"]
    for name in ("events_processed", "cycles_completed"):
        if name in metrics_payload:
            summary_metrics[name] = metrics_payload[name]
    active_models = LIFECYCLE_STATE["active_models"]
    if event.active_classification_model_version:
        active_models["classification"] = event.active_classification_model_version
    if event.active_localization_model_version:
        active_models["localization"] = event.active_localization_model_version
    if event.candidate_initial_model_version and not active_models:
        active_models.setdefault("classification", event.candidate_initial_model_version)
        active_models.setdefault("localization", event.candidate_initial_model_version)
    _record_lifecycle_final_models(event)


def _lifecycle_gate_labels(
    event: LifecycleEventRequest,
    *,
    role: str,
    model: str,
    metric_name: str,
) -> tuple[tuple[str, Any], ...]:
    return (
        ("scenario_id", event.scenario_id),
        ("lifecycle_run_id", event.lifecycle_run_id),
        ("cycle_id", event.cycle_id or "unknown"),
        ("role", role),
        ("model", model),
        ("metric", metric_name),
    )


def _lifecycle_gate_delta_labels(
    event: LifecycleEventRequest,
    *,
    role: str,
    metric_name: str,
) -> tuple[tuple[str, Any], ...]:
    return (
        ("scenario_id", event.scenario_id),
        ("lifecycle_run_id", event.lifecycle_run_id),
        ("cycle_id", event.cycle_id or "unknown"),
        ("role", role),
        ("metric", metric_name),
    )


def _record_lifecycle_final_models(event: LifecycleEventRequest) -> None:
    final_models = LIFECYCLE_STATE["final_models"]
    specs = (
        (
            "classification",
            event.active_classification_model_version,
            event.active_classification_registered_model_name,
            event.active_classification_registered_model_version,
        ),
        (
            "localization",
            event.active_localization_model_version,
            event.active_localization_registered_model_name,
            event.active_localization_registered_model_version,
        ),
    )
    for role, version, model_name, model_version in specs:
        if not version:
            continue
        existing = final_models.get(role, {})
        final_models[role] = {
            "version": version,
            "registered_model_name": model_name or existing.get("registered_model_name", ""),
            "registered_model_version": model_version or existing.get("registered_model_version", ""),
        }


def _record_lifecycle_promotion_status(event: LifecycleEventRequest) -> None:
    for role, status in (
        ("localization", event.localization_promotion_status),
        ("classification", event.classification_promotion_status),
    ):
        if not status:
            continue
        key = (event.lifecycle_run_id, event.cycle_id or "unknown", role, status)
        seen: set[tuple[str, str, str, str]] = LIFECYCLE_STATE["promotion_seen"]
        if key not in seen:
            seen.add(key)
            counters = LIFECYCLE_STATE["promotion_counters"]
            labels = (
                ("scenario_id", event.scenario_id),
                ("lifecycle_run_id", event.lifecycle_run_id),
                ("role", role),
                ("status", status),
            )
            counters[labels] = counters.get(labels, 0) + 1
        if status == "promoted" and event.candidate_version:
            LIFECYCLE_STATE["active_models"][role] = event.candidate_version
        decision_labels = (
            ("scenario_id", event.scenario_id),
            ("lifecycle_run_id", event.lifecycle_run_id),
            ("cycle_id", event.cycle_id or "unknown"),
            ("role", role),
            ("status", status),
            ("candidate_version", event.candidate_version or ""),
        )
        LIFECYCLE_STATE["promotion_decisions"][decision_labels] = 1


def _reject_sensitive_drift_payload(payload: dict[str, Any]) -> None:
    def visit(value: Any, *, key: str = "") -> None:
        lowered_key = key.lower()
        if any(marker in lowered_key for marker in LIFECYCLE_SENSITIVE_KEYS):
            _raise_api_error(
                status_code=422,
                error_code="sensitive_drift_field",
                message="Drift event contains a sensitive or non-observable field.",
                reason=f"Drift field {key!r} is not accepted by the metrics API.",
            )
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                if key == "metrics" and child_key not in DRIFT_ALLOWED_METRICS:
                    _raise_api_error(
                        status_code=422,
                        error_code="unsupported_drift_metric",
                        message="Drift event contains an unsupported metric.",
                        reason=f"Metric {child_key!r} is not accepted by the metrics API.",
                    )
                visit(child_value, key="" if key == "metrics" else str(child_key))
            return
        if isinstance(value, list):
            for item in value:
                visit(item, key=key)
            return
        if isinstance(value, str):
            text = value.lower()
            if (
                "://" in text
                or ":\\" in text
                or "\\\\" in text
                or "/.cache/" in text
                or "\\.cache\\" in text
                or "/data/" in text
                or "\\data\\" in text
                or "/models/" in text
                or "\\models\\" in text
            ):
                _raise_api_error(
                    status_code=422,
                    error_code="sensitive_drift_value",
                    message="Drift event contains a sensitive value.",
                    reason=f"Drift value for {key!r} looks like a path or URI.",
                )

    visit(payload)


def _record_drift_observation(event: DriftEventRequest) -> None:
    payload = event.model_dump(mode="json", exclude_none=True)
    _reject_sensitive_drift_payload(payload)
    status = event.status if event.status in DRIFT_ALLOWED_STATUSES else "clear"
    metrics_payload = {
        key: float(value)
        for key, value in (payload.get("metrics") or {}).items()
        if key in DRIFT_ALLOWED_METRICS and _finite_metric(value) is not None
    }
    if event.window_events is not None:
        metrics_payload["window_events"] = float(event.window_events)
    if event.window_index is not None:
        metrics_payload["window_index"] = float(event.window_index)
    if event.first_confirmed_window_index is not None:
        metrics_payload["first_confirmed_window_index"] = float(event.first_confirmed_window_index)
    active_models: dict[str, dict[str, str]] = {}
    for role, model_payload in (event.active_models or {}).items():
        if role not in DRIFT_ALLOWED_MODEL_ROLES:
            continue
        active_models[role] = {
            key: str(value)
            for key, value in model_payload.items()
            if key in DRIFT_ACTIVE_MODEL_ALLOWED_FIELDS and value not in {None, ""}
        }
    DRIFT_STATE["current"] = {
        "event_type": event.event_type,
        "scenario_id": event.scenario_id,
        "status": status,
        "source_domain": event.source_domain,
        "lifecycle_run_id": event.lifecycle_run_id or "",
        "cycle_id": event.cycle_id or "",
        "updated_at": time.time(),
        "trigger_lifecycle": bool(event.trigger_lifecycle),
        "active_models": active_models,
        "metrics": metrics_payload,
    }


@app.post("/internal/lifecycle/events")
def record_lifecycle_metric_event(
    request: LifecycleEventRequest,
    x_iqa_service_token: str | None = Header(default=None, alias="X-IQA-Service-Token"),
) -> dict[str, Any]:
    _require_token("IQA_SERVICE_TOKEN", x_iqa_service_token)
    _record_lifecycle_observation(request)
    return {
        "accepted": True,
        "event_type": request.event_type,
        "scenario_id": request.scenario_id,
        "lifecycle_run_id": request.lifecycle_run_id,
    }


@app.post("/internal/drift/events")
def record_drift_metric_event(
    request: DriftEventRequest,
    x_iqa_service_token: str | None = Header(default=None, alias="X-IQA-Service-Token"),
) -> dict[str, Any]:
    _require_token("IQA_SERVICE_TOKEN", x_iqa_service_token)
    _record_drift_observation(request)
    return {
        "accepted": True,
        "event_type": request.event_type,
        "scenario_id": request.scenario_id,
        "status": request.status,
    }


def _lifecycle_metrics_lines() -> list[str]:
    current = dict(LIFECYCLE_STATE["current"])
    lines = [
        "# HELP iqa_lifecycle_cycle_current Current IQA lifecycle cycle observed by the API",
        "# TYPE iqa_lifecycle_cycle_current gauge",
        "# HELP iqa_lifecycle_epoch_current Current IQA lifecycle training epoch observed by the API",
        "# TYPE iqa_lifecycle_epoch_current gauge",
        "# HELP iqa_lifecycle_epoch_pixel_aupimo Latest lifecycle epoch pixel AUPIMO",
        "# TYPE iqa_lifecycle_epoch_pixel_aupimo gauge",
        "# HELP iqa_lifecycle_epoch_pixel_ap Latest lifecycle epoch pixel AP",
        "# TYPE iqa_lifecycle_epoch_pixel_ap gauge",
        "# HELP iqa_lifecycle_epoch_image_ap Latest lifecycle epoch image AP",
        "# TYPE iqa_lifecycle_epoch_image_ap gauge",
        "# HELP iqa_lifecycle_epoch_metric Latest lifecycle epoch metric by metric label",
        "# TYPE iqa_lifecycle_epoch_metric gauge",
        "# HELP iqa_lifecycle_gate_metric_delta Latest lifecycle gate metric delta by role",
        "# TYPE iqa_lifecycle_gate_metric_delta gauge",
        "# HELP iqa_lifecycle_gate_fn_delta Latest lifecycle classification false negative delta",
        "# TYPE iqa_lifecycle_gate_fn_delta gauge",
        "# HELP iqa_lifecycle_gate_value Lifecycle gate active/candidate metric value",
        "# TYPE iqa_lifecycle_gate_value gauge",
        "# HELP iqa_lifecycle_gate_delta Lifecycle gate metric delta by role and metric",
        "# TYPE iqa_lifecycle_gate_delta gauge",
        "# HELP iqa_lifecycle_promotion_total IQA lifecycle promotion decisions by role and status",
        "# TYPE iqa_lifecycle_promotion_total counter",
        "# HELP iqa_lifecycle_promotion_decision_info Latest lifecycle promotion decisions by role",
        "# TYPE iqa_lifecycle_promotion_decision_info gauge",
        "# HELP iqa_lifecycle_active_model_info Active lifecycle model versions observed by the API",
        "# TYPE iqa_lifecycle_active_model_info gauge",
        "# HELP iqa_lifecycle_final_model_info Final promoted model versions for the lifecycle run",
        "# TYPE iqa_lifecycle_final_model_info gauge",
        "# HELP iqa_lifecycle_run_events_processed Lifecycle run events processed",
        "# TYPE iqa_lifecycle_run_events_processed gauge",
        "# HELP iqa_lifecycle_run_cycles_completed Lifecycle run cycles completed",
        "# TYPE iqa_lifecycle_run_cycles_completed gauge",
    ]
    if current:
        base_labels = _lifecycle_base_labels(current)
        lines.append(f"iqa_lifecycle_cycle_current{{{_metric_labels(base_labels)}}} {_cycle_number(current.get('cycle_id'))}")
        epoch_recent = _observability_is_recent(LIFECYCLE_STATE.get("epoch_updated_at"))
        if current.get("epoch") is not None and epoch_recent:
            lines.append(f"iqa_lifecycle_epoch_current{{{_metric_labels(base_labels)}}} {int(current['epoch'])}")
        epoch_metrics = LIFECYCLE_STATE["epoch_metrics"] if epoch_recent else {}
        for metric_name, value in sorted(epoch_metrics.items()):
            finite_value = _finite_metric(value)
            if finite_value is not None:
                labels = base_labels + (("metric", metric_name),)
                lines.append(f"iqa_lifecycle_epoch_metric{{{_metric_labels(labels)}}} {finite_value}")
        epoch_specs = (
            ("pixel_aupimo", "iqa_lifecycle_epoch_pixel_aupimo"),
            ("pixel_ap", "iqa_lifecycle_epoch_pixel_ap"),
            ("image_ap", "iqa_lifecycle_epoch_image_ap"),
        )
        for source_name, metric_name in epoch_specs:
            value = _finite_metric(epoch_metrics.get(source_name))
            if value is not None:
                lines.append(f"{metric_name}{{{_metric_labels(base_labels)}}} {value}")
        for role, role_metrics in sorted(LIFECYCLE_STATE["gate_metrics"].items()):
            labels = base_labels + (("role", role),)
            metric_delta = _finite_metric(role_metrics.get("metric_delta"))
            if metric_delta is not None:
                lines.append(f"iqa_lifecycle_gate_metric_delta{{{_metric_labels(labels)}}} {metric_delta}")
            fn_delta = _finite_metric(role_metrics.get("fn_delta"))
            if fn_delta is not None:
                lines.append(f"iqa_lifecycle_gate_fn_delta{{{_metric_labels(labels)}}} {fn_delta}")
        for labels, value in sorted(LIFECYCLE_STATE["gate_values"].items(), key=lambda item: str(item[0])):
            finite_value = _finite_metric(value)
            if finite_value is not None:
                lines.append(f"iqa_lifecycle_gate_value{{{_metric_labels(labels)}}} {finite_value}")
        for labels, value in sorted(LIFECYCLE_STATE["gate_deltas"].items(), key=lambda item: str(item[0])):
            finite_value = _finite_metric(value)
            if finite_value is not None:
                lines.append(f"iqa_lifecycle_gate_delta{{{_metric_labels(labels)}}} {finite_value}")
        for role, version in sorted(LIFECYCLE_STATE["active_models"].items()):
            labels = (
                ("scenario_id", current.get("scenario_id")),
                ("role", role),
                ("version", version),
                ("run_id", current.get("lifecycle_run_id")),
                ("cycle_id", current.get("cycle_id")),
            )
            lines.append(f"iqa_lifecycle_active_model_info{{{_metric_labels(labels)}}} 1")
        for role, model_payload in sorted(LIFECYCLE_STATE["final_models"].items()):
            labels = (
                ("scenario_id", current.get("scenario_id")),
                ("lifecycle_run_id", current.get("lifecycle_run_id")),
                ("role", role),
                ("version", model_payload.get("version", "")),
                ("registered_model_version", model_payload.get("registered_model_version", "")),
                ("registered_model_name", model_payload.get("registered_model_name", "")),
            )
            lines.append(f"iqa_lifecycle_final_model_info{{{_metric_labels(labels)}}} 1")
        summary_metrics = LIFECYCLE_STATE["summary_metrics"]
        events_processed = _finite_metric(summary_metrics.get("events_processed"))
        if events_processed is not None:
            lines.append(f"iqa_lifecycle_run_events_processed{{{_metric_labels(base_labels)}}} {events_processed}")
        cycles_completed = _finite_metric(summary_metrics.get("cycles_completed"))
        if cycles_completed is not None:
            lines.append(f"iqa_lifecycle_run_cycles_completed{{{_metric_labels(base_labels)}}} {cycles_completed}")
    for labels, value in sorted(LIFECYCLE_STATE["promotion_counters"].items(), key=lambda item: str(item[0])):
        lines.append(f"iqa_lifecycle_promotion_total{{{_metric_labels(labels)}}} {value}")
    for labels, value in sorted(LIFECYCLE_STATE["promotion_decisions"].items(), key=lambda item: str(item[0])):
        lines.append(f"iqa_lifecycle_promotion_decision_info{{{_metric_labels(labels)}}} {value}")
    return lines


def _drift_metrics_lines() -> list[str]:
    current = dict(DRIFT_STATE["current"])
    lines = [
        "# HELP iqa_drift_score Current IQA drift score received by the API",
        "# TYPE iqa_drift_score gauge",
        "# HELP iqa_drift_status Current IQA drift status as a one-hot gauge",
        "# TYPE iqa_drift_status gauge",
        "# HELP iqa_drift_window_events Number of events in the current drift window",
        "# TYPE iqa_drift_window_events gauge",
        "# HELP iqa_drift_window_index Current drift observation window index",
        "# TYPE iqa_drift_window_index gauge",
        "# HELP iqa_drift_first_confirmed_window First window index where drift was confirmed",
        "# TYPE iqa_drift_first_confirmed_window gauge",
        "# HELP iqa_drift_alert_rate Alert decision rate in the current drift window",
        "# TYPE iqa_drift_alert_rate gauge",
        "# HELP iqa_drift_red_rate Red decision rate in the current drift window",
        "# TYPE iqa_drift_red_rate gauge",
        "# HELP iqa_drift_unexpected_red_rate Red decision rate on conforming pieces in the current drift window",
        "# TYPE iqa_drift_unexpected_red_rate gauge",
        "# HELP iqa_drift_roi_fail_rate ROI failure rate in the current drift window",
        "# TYPE iqa_drift_roi_fail_rate gauge",
        "# HELP iqa_drift_oracle_fn_rate Oracle false-negative rate in the current drift window",
        "# TYPE iqa_drift_oracle_fn_rate gauge",
        "# HELP iqa_drift_domain_ratio Source-domain ratio in the current drift window",
        "# TYPE iqa_drift_domain_ratio gauge",
        "# HELP iqa_drift_degradation_score Model degradation score independent of source-domain ratio",
        "# TYPE iqa_drift_degradation_score gauge",
        "# HELP iqa_drift_domain_score Source-domain drift score component",
        "# TYPE iqa_drift_domain_score gauge",
        "# HELP iqa_drift_trigger_lifecycle Whether the observed drift window should trigger lifecycle correction",
        "# TYPE iqa_drift_trigger_lifecycle gauge",
        "# HELP iqa_drift_active_model_info Active models used by the drift observation replay",
        "# TYPE iqa_drift_active_model_info gauge",
    ]
    if not current:
        return lines
    if not _observability_is_recent(current.get("updated_at")):
        return lines

    scenario_id = current.get("scenario_id")
    source_domain = current.get("source_domain") or "piece_a_p4"
    status = current.get("status") if current.get("status") in DRIFT_ALLOWED_STATUSES else "clear"
    metrics_payload = current.get("metrics") or {}
    base_labels = (("scenario_id", scenario_id), ("source_domain", source_domain))
    for status_name in ("clear", "suspected", "confirmed"):
        labels = base_labels + (("status", status_name),)
        lines.append(f"iqa_drift_status{{{_metric_labels(labels)}}} {1 if status == status_name else 0}")
    lines.append(
        f"iqa_drift_trigger_lifecycle{{{_metric_labels(base_labels)}}} "
        f"{1 if current.get('trigger_lifecycle') else 0}"
    )
    metric_specs = (
        ("drift_score", "iqa_drift_score"),
        ("window_events", "iqa_drift_window_events"),
        ("window_index", "iqa_drift_window_index"),
        ("first_confirmed_window_index", "iqa_drift_first_confirmed_window"),
        ("alert_rate", "iqa_drift_alert_rate"),
        ("red_rate", "iqa_drift_red_rate"),
        ("unexpected_red_rate", "iqa_drift_unexpected_red_rate"),
        ("roi_fail_rate", "iqa_drift_roi_fail_rate"),
        ("oracle_fn_rate", "iqa_drift_oracle_fn_rate"),
        ("domain_ratio", "iqa_drift_domain_ratio"),
        ("domain_score", "iqa_drift_domain_score"),
        ("degradation_score", "iqa_drift_degradation_score"),
    )
    for source_name, metric_name in metric_specs:
        value = _finite_metric(metrics_payload.get(source_name))
        if value is not None:
            lines.append(f"{metric_name}{{{_metric_labels(base_labels)}}} {value}")
    for role, model_payload in sorted((current.get("active_models") or {}).items()):
        labels = base_labels + (
            ("role", role),
            ("version", model_payload.get("version", "")),
            ("registry_model_name", model_payload.get("registry_model_name", "")),
            ("registered_model_version", model_payload.get("registered_model_version", "")),
            ("registry_stage", model_payload.get("registry_stage", "")),
            ("runtime_contract_status", model_payload.get("runtime_contract_status", "")),
        )
        lines.append(f"iqa_drift_active_model_info{{{_metric_labels(labels)}}} 1")
    return lines


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
    lines.extend(_drift_metrics_lines())
    lines.extend(_lifecycle_metrics_lines())
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
    "DRIFT_STATE",
    "LIFECYCLE_STATE",
    "METADATA_WRITE_THROUGH",
    "app",
    "feedback",
    "health",
    "metrics",
    "model_version",
    "predict",
    "predict_piece_event",
    "record_drift_metric_event",
    "record_lifecycle_metric_event",
    "reload_model",
    "next_replay_event",
    "reset_replay_run",
    "replay_scenarios",
]
