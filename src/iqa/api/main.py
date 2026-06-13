"""FastAPI gateway for IQA."""

from __future__ import annotations

import json
import os
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
    inference_result = placeholder_inference(
        InferenceRequest(
            piece_event_id=request.piece_event_id,
            scenario_id=request.scenario_id,
            image_uri=request.image_uri,
        )
    )

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
        raise HTTPException(status_code=404, detail="Unknown prediction_id.")

    if prediction["piece_event_id"] != request.piece_event_id:
        raise HTTPException(status_code=409, detail="prediction_id does not match piece_event_id.")

    if prediction["scenario_id"] != request.scenario_id:
        raise HTTPException(status_code=409, detail="prediction_id does not match scenario_id.")

    if prediction.get("feedback_closed") is True:
        raise HTTPException(status_code=409, detail="Prediction already has a closed feedback.")

    return prediction


@app.post("/feedback")
def feedback(
    request: FeedbackRequest,
    x_iqa_service_token: str | None = Header(default=None, alias="X-IQA-Service-Token"),
) -> dict[str, Any]:
    _require_token("IQA_SERVICE_TOKEN", x_iqa_service_token)

    prediction = _get_open_prediction_for_feedback(request)

    if request.feedback_source != "oracle_gt":
        return {
            "accepted": False,
            "prediction_id": request.prediction_id,
            "feedback_closed": False,
            "reason": "MVP accepts only oracle_gt feedback; human_sophie is a future interface.",
        }

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

    FEEDBACK_STORE[request.prediction_id] = {
        "prediction_id": request.prediction_id,
        "piece_event_id": request.piece_event_id,
        "scenario_id": request.scenario_id,
        "feedback_source": request.feedback_source,
        "feedback_closed": True,
        "closed_at": closed_at,
        "verdict": verdict.to_dict(),
    }

    return {
        "accepted": True,
        "prediction_id": request.prediction_id,
        "feedback_closed": True,
        "feedback": verdict.to_dict(),
    }


@app.get("/metrics")
def metrics() -> str:
    return "# HELP iqa_api_up IQA API availability\n# TYPE iqa_api_up gauge\niqa_api_up 1\n"


@app.post("/admin/reload-model")
def reload_model(
    request: ReloadModelRequest,
    x_iqa_admin_token: str | None = Header(default=None, alias="X-IQA-Admin-Token"),
) -> dict[str, Any]:
    _require_token("IQA_ADMIN_TOKEN", x_iqa_admin_token)
    model_name = registered_model_name(request.scenario_id)
    return {
        "accepted": True,
        "source_of_truth": "mlflow_registry",
        "target": ModelRegistryRef(
            scenario_id=request.scenario_id,
            registered_model_name=model_name,
            stage=request.stage,
        ).to_dict(),
    }


__all__ = [
    "FeedbackRequest",
    "PieceEventPredictRequest",
    "PredictRequest",
    "ReloadModelRequest",
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
