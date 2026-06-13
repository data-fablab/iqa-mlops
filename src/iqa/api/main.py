"""FastAPI gateway for IQA."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

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
    return {
        "service": "iqa-api",
        "delegated_to": "iqa-inference",
        "prediction": inference_result.to_dict(),
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


@app.post("/feedback")
def feedback(
    request: FeedbackRequest,
    x_iqa_service_token: str | None = Header(default=None, alias="X-IQA-Service-Token"),
) -> dict[str, Any]:
    _require_token("IQA_SERVICE_TOKEN", x_iqa_service_token)
    if request.feedback_source != "oracle_gt":
        return {
            "accepted": False,
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
    return {"accepted": True, "feedback": verdict.to_dict()}


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
