"""FastAPI service boundary for IQA PyTorch inference."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from iqa.inference.contracts import InferenceRequest, placeholder_inference
from iqa.inference.prediction_journal import append_journal
from iqa.inference.real_inference import get_scorer, real_inference_enabled
from iqa.runtime import gpu_lock


def _active_covered_classes() -> list[str]:
    """Read covered_classes from the active PatchCore manifest (cheap, scrape-time).

    Reads the JSON manifest directly instead of forcing the detector (bank +
    backbone) to load on every scrape. Degrades to an empty list on any error so
    /metrics never fails.
    """

    if not real_inference_enabled():
        return []
    try:
        manifest_path = Path(get_scorer().domain_drift_dir) / "model_manifest.json"
        if not manifest_path.exists():
            return []
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        classes = payload.get("covered_classes") or []
        return [str(c) for c in classes]
    except Exception:  # noqa: BLE001 - /metrics must never 500
        return []


def _demo_hold_enabled() -> bool:
    return os.environ.get("IQA_GPU_DEMO_HOLD", "").strip().lower() in {"1", "true", "yes", "on"}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Hold the GPU lock for the whole demo when ``IQA_GPU_DEMO_HOLD`` is set.

    This guarantees no ``iqa-trainer`` can grab the single GPU while the live
    inference demo is running. Acquire is blocking: the demo waits for any
    in-flight training run to release the GPU before serving.
    """

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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "iqa-inference"}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    gpu_lock_held = 1 if getattr(app.state, "gpu_lock_held", False) else 0
    covered = _active_covered_classes()
    lines = [
        "# HELP iqa_inference_up IQA inference availability",
        "# TYPE iqa_inference_up gauge",
        "iqa_inference_up 1",
        "# HELP iqa_inference_gpu_lock_held IQA inference demo holds the single-GPU lock",
        "# TYPE iqa_inference_gpu_lock_held gauge",
        f"iqa_inference_gpu_lock_held {gpu_lock_held}",
        "# HELP iqa_domain_drift_covered_classes Number of classes covered by the active PatchCore bank",
        "# TYPE iqa_domain_drift_covered_classes gauge",
        f"iqa_domain_drift_covered_classes {len(covered)}",
    ]
    if covered:
        # Info-style metric: the class list rides on a label so the demo can show
        # the cumulative coverage [class1, class2, class3] in Grafana.
        classes_label = ",".join(covered).replace("\\", "").replace('"', "")
        lines += [
            "# HELP iqa_domain_drift_covered_classes_info Classes covered by the active PatchCore bank (label)",
            "# TYPE iqa_domain_drift_covered_classes_info gauge",
            f'iqa_domain_drift_covered_classes_info{{classes="{classes_label}"}} 1',
        ]
    return "\n".join(lines) + "\n"


@app.post("/predict")
def predict(request: InferenceServiceRequest) -> dict[str, str | float | None]:
    inference_request = InferenceRequest(
        piece_event_id=request.piece_event_id,
        scenario_id=request.scenario_id,
        image_uri=request.image_uri,
    )
    if real_inference_enabled():
        try:
            result = get_scorer().predict(inference_request).to_dict()
        except Exception:  # noqa: BLE001 - never 500 the demo; degrade to placeholder
            result = placeholder_inference(inference_request).to_dict()
    else:
        result = placeholder_inference(inference_request).to_dict()
    append_journal(result)
    return result


@app.post("/reload-model")
def reload_model(checkpoint_path: str | None = None) -> dict[str, str | None]:
    """Drop the cached model so the next prediction loads a fresh checkpoint.

    Called after a retrain promotes a new Feature-AE so recovery to Vert reflects
    the updated model. Optional ``checkpoint_path`` switches the active checkpoint.
    """
    scorer = get_scorer()
    scorer.reload(checkpoint_path)
    return {"status": "reloaded", "checkpoint_path": scorer.checkpoint_path, "feature_ae_version": scorer.feature_ae_version}


__all__ = ["InferenceServiceRequest", "app", "health", "lifespan", "metrics", "predict"]
