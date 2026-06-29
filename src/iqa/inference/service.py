"""FastAPI service boundary for IQA PyTorch inference."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from iqa.inference.contracts import InferenceRequest, InferenceResult
from iqa.inference.runtime import run_inference_pipeline
from iqa.runtime import gpu_lock


def _demo_hold_enabled() -> bool:
    return os.environ.get("IQA_GPU_DEMO_HOLD", "").strip().lower() in {"1", "true", "yes", "on"}


def _inference_device() -> str:
    return os.environ.get("IQA_INFERENCE_DEVICE", "cpu").strip() or "cpu"


def _run_real_inference(request: InferenceRequest) -> InferenceResult:
    return run_inference_pipeline(
        request,
        device=_inference_device(),
    ).result


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Hold the GPU lock for the whole demo when requested."""

    if _demo_hold_enabled():
        with gpu_lock(owner="iqa-inference-demo", blocking=True):
            app.state.gpu_lock_held = True
            yield
        app.state.gpu_lock_held = False
    else:
        app.state.gpu_lock_held = False
        yield


app = FastAPI(
    title="Industrial Quality Assistant Inference",
    version="0.1.0",
    lifespan=lifespan,
)


class InferenceServiceRequest(BaseModel):
    piece_event_id: str
    scenario_id: str = "production_replay_natural"
    image_uri: str
    sha256: str | None = None
    lot_id: str | None = None
    source_class: str | None = None
    dataset_version: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "iqa-inference"}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    gpu_lock_held = 1 if getattr(app.state, "gpu_lock_held", False) else 0
    lines = [
        "# HELP iqa_inference_up IQA inference availability",
        "# TYPE iqa_inference_up gauge",
        "iqa_inference_up 1",
        "# HELP iqa_inference_gpu_lock_held IQA inference demo holds the single-GPU lock",
        "# TYPE iqa_inference_gpu_lock_held gauge",
        f"iqa_inference_gpu_lock_held {gpu_lock_held}",
    ]
    return "\n".join(lines) + "\n"


@app.post("/predict")
def predict(request: InferenceServiceRequest) -> dict[str, str | float | None]:
    inference_request = InferenceRequest(
        piece_event_id=request.piece_event_id,
        scenario_id=request.scenario_id,
        image_uri=request.image_uri,
        sha256=request.sha256,
        lot_id=request.lot_id,
        source_class=request.source_class,
        dataset_version=request.dataset_version,
    )

    try:
        result = _run_real_inference(inference_request)
    except FileNotFoundError as error:
        detail = str(error)
        if detail.startswith("Input image not found"):
            raise HTTPException(status_code=404, detail=detail) from error
        raise HTTPException(
            status_code=503,
            detail=f"Inference unavailable: {detail}",
        ) from error
    except ValueError as error:
        detail = str(error)
        if "Input image checksum mismatch" in detail:
            raise HTTPException(status_code=422, detail=detail) from error
        raise HTTPException(
            status_code=503,
            detail=f"Inference unavailable: {detail}",
        ) from error
    except (ImportError, RuntimeError, OSError) as error:
        raise HTTPException(
            status_code=503,
            detail=f"Inference unavailable: {error}",
        ) from error

    return result.to_dict()


__all__ = [
    "InferenceServiceRequest",
    "app",
    "health",
    "lifespan",
    "metrics",
    "predict",
]
