"""FastAPI service boundary for IQA PyTorch inference."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from iqa.inference.contracts import InferenceRequest, placeholder_inference


app = FastAPI(title="Industrial Quality Assistant Inference", version="0.1.0")


class InferenceServiceRequest(BaseModel):
    piece_event_id: str
    scenario_id: str = "production_replay_natural"
    image_uri: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "iqa-inference"}


@app.get("/metrics")
def metrics() -> str:
    return "# HELP iqa_inference_up IQA inference availability\n# TYPE iqa_inference_up gauge\niqa_inference_up 1\n"


@app.post("/predict")
def predict(request: InferenceServiceRequest) -> dict[str, str | float | None]:
    result = placeholder_inference(
        InferenceRequest(
            piece_event_id=request.piece_event_id,
            scenario_id=request.scenario_id,
            image_uri=request.image_uri,
        )
    )
    return result.to_dict()


__all__ = ["InferenceServiceRequest", "app", "health", "metrics", "predict"]
