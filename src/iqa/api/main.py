"""FastAPI skeleton for IQA Phase 1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parents[3]
ROI_MANIFEST = BASE_DIR / "models" / "manifests" / "roi_segmenter_v001_fixed" / "model_manifest.json"
FEATURE_AE_MANIFEST = BASE_DIR / "models" / "manifests" / "rd_feature_ae_gated_v001_bootstrap" / "model_manifest.json"

app = FastAPI(title="Industrial Quality Assistant API", version="0.1.0")


class PredictRequest(BaseModel):
    piece_event_id: str
    scenario_id: str | None = None
    image_uri: str = Field(..., description="S3/DVC/local URI for the primary image.")


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": "missing", "manifest_path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "iqa-api"}


@app.get("/model/version")
def model_version() -> dict[str, Any]:
    return {
        "roi_segmenter": _read_manifest(ROI_MANIFEST),
        "feature_ae": _read_manifest(FEATURE_AE_MANIFEST),
    }


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    raise HTTPException(
        status_code=501,
        detail={
            "message": "Phase 1 placeholder: inference runtime is not wired yet.",
            "piece_event_id": request.piece_event_id,
        },
    )


__all__ = ["PredictRequest", "app", "health", "model_version", "predict"]
