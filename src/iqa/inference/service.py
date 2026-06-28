"""FastAPI boundary for IQA inference and model reload."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from iqa.inference.contracts import (
    InferenceRequest,
    InferenceResult,
)
from iqa.inference.model_loader import (
    LoadedModel,
    ProdModelLoader,
)
from iqa.inference.runtime import (
    run_inference_pipeline,
)
from iqa.models.artifacts import (
    DEFAULT_FEATURE_AE_MODEL_VERSION,
    DEFAULT_ROI_MODEL_VERSION,
)
from iqa.runtime import gpu_lock


@dataclass(frozen=True)
class ActiveInferenceRuntime:
    """Immutable snapshot used by one prediction."""

    scenario_id: str
    feature_ae_version: str
    roi_model_version: str
    registry_version: str = ""
    model_uri: str = ""
    model_id: str = ""
    checkpoint_path: Path | None = None
    decision_thresholds: dict[str, Any] | None = None
    reference_contract: Any | None = None
    loaded_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "feature_ae_version": (
                self.feature_ae_version
            ),
            "roi_model_version": (
                self.roi_model_version
            ),
            "registry_version": (
                self.registry_version
            ),
            "model_uri": self.model_uri,
            "model_id": self.model_id,
            "checkpoint_path": (
                str(self.checkpoint_path)
                if self.checkpoint_path
                else None
            ),
            "loaded_at": self.loaded_at,
        }


def _configured_scenario() -> str:
    return (
        os.environ.get(
            "IQA_INFERENCE_SCENARIO_ID",
            "production_replay_natural",
        ).strip()
        or "production_replay_natural"
    )


_RUNTIME_LOCK = RLock()
_ACTIVE_RUNTIME = ActiveInferenceRuntime(
    scenario_id=_configured_scenario(),
    feature_ae_version=(
        DEFAULT_FEATURE_AE_MODEL_VERSION
    ),
    roi_model_version=DEFAULT_ROI_MODEL_VERSION,
)


def _runtime_snapshot() -> ActiveInferenceRuntime:
    with _RUNTIME_LOCK:
        return _ACTIVE_RUNTIME


def _swap_runtime(
    runtime: ActiveInferenceRuntime,
) -> None:
    global _ACTIVE_RUNTIME

    with _RUNTIME_LOCK:
        _ACTIVE_RUNTIME = runtime


def _runtime_from_loaded_model(
    loaded: LoadedModel,
) -> ActiveInferenceRuntime:
    if loaded.checkpoint_path is None:
        raise ValueError(
            "loaded_model_missing_checkpoint_path"
        )
    if loaded.reference_contract is None:
        raise ValueError(
            "loaded_model_missing_reference_contract"
        )
    if not loaded.decision_thresholds:
        raise ValueError(
            "loaded_model_missing_decision_thresholds"
        )

    return ActiveInferenceRuntime(
        scenario_id=loaded.scenario_id,
        feature_ae_version=(
            loaded.feature_ae_version
        ),
        roi_model_version=(
            loaded.roi_model_version
        ),
        registry_version=loaded.version,
        model_uri=loaded.artifact_uri,
        model_id=loaded.model_id,
        checkpoint_path=loaded.checkpoint_path,
        decision_thresholds=dict(
            loaded.decision_thresholds
        ),
        reference_contract=(
            loaded.reference_contract
        ),
        loaded_at=datetime.now(
            UTC
        ).isoformat(),
    )


def _demo_hold_enabled() -> bool:
    return (
        os.environ.get(
            "IQA_GPU_DEMO_HOLD",
            "",
        )
        .strip()
        .lower()
        in {"1", "true", "yes", "on"}
    )


def _inference_device() -> str:
    return (
        os.environ.get(
            "IQA_INFERENCE_DEVICE",
            "cpu",
        ).strip()
        or "cpu"
    )


def _tracking_uri() -> str | None:
    return (
        os.environ.get("MLFLOW_TRACKING_URI")
        or os.environ.get(
            "IQA_MLFLOW_TRACKING_URI"
        )
    )


def _run_real_inference(
    request: InferenceRequest,
) -> InferenceResult:
    runtime = _runtime_snapshot()

    return run_inference_pipeline(
        request,
        device=_inference_device(),
        roi_model_version=(
            runtime.roi_model_version
        ),
        feature_ae_version=(
            runtime.feature_ae_version
        ),
        feature_checkpoint=(
            runtime.checkpoint_path
        ),
        decision_thresholds=(
            runtime.decision_thresholds
        ),
        feature_ae_reference_contract=(
            runtime.reference_contract
        ),
    ).result


@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> AsyncIterator[None]:
    if _demo_hold_enabled():
        with gpu_lock(
            owner="iqa-inference-demo",
            blocking=True,
        ):
            app.state.gpu_lock_held = True
            yield
        app.state.gpu_lock_held = False
    else:
        app.state.gpu_lock_held = False
        yield


app = FastAPI(
    title="Industrial Quality Assistant Inference",
    version="0.2.0",
    lifespan=lifespan,
)


class InferenceServiceRequest(BaseModel):
    piece_event_id: str
    scenario_id: str = (
        "production_replay_natural"
    )
    image_uri: str
    sha256: str | None = None
    lot_id: str | None = None
    source_class: str | None = None
    dataset_version: str | None = None


class ReloadInferenceRequest(BaseModel):
    scenario_id: str = (
        "production_replay_natural"
    )
    stage: str = "prod"


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "iqa-inference",
    }


@app.get("/model/version")
def model_version() -> dict[str, Any]:
    return _runtime_snapshot().to_dict()


@app.get("/metrics")
def metrics() -> str:
    gpu_lock_held = (
        1
        if getattr(
            app.state,
            "gpu_lock_held",
            False,
        )
        else 0
    )
    lines = [
        "# HELP iqa_inference_up "
        "IQA inference availability",
        "# TYPE iqa_inference_up gauge",
        "iqa_inference_up 1",
        "# HELP iqa_inference_gpu_lock_held "
        "IQA inference demo holds the "
        "single-GPU lock",
        "# TYPE "
        "iqa_inference_gpu_lock_held gauge",
        "iqa_inference_gpu_lock_held "
        f"{gpu_lock_held}",
    ]
    return "\n".join(lines) + "\n"


@app.post("/admin/reload-model")
def reload_model(
    request: ReloadInferenceRequest,
    x_iqa_admin_token: str | None = Header(
        default=None,
        alias="X-IQA-Admin-Token",
    ),
) -> dict[str, Any]:
    expected_token = (
        os.environ.get(
            "IQA_INFERENCE_RELOAD_TOKEN"
        )
        or os.environ.get("IQA_ADMIN_TOKEN")
    )

    if not expected_token:
        raise HTTPException(
            status_code=503,
            detail=(
                "Inference reload token "
                "is not configured."
            ),
        )
    if x_iqa_admin_token != expected_token:
        raise HTTPException(
            status_code=401,
            detail=(
                "Missing or invalid "
                "inference reload token."
            ),
        )
    if request.stage != "prod":
        raise HTTPException(
            status_code=422,
            detail=(
                "Only the prod alias "
                "can be reloaded."
            ),
        )
    if (
        request.scenario_id
        != _configured_scenario()
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                "This inference service "
                "serves one configured scenario."
            ),
        )

    previous = _runtime_snapshot()

    try:
        loaded = ProdModelLoader(
            request.scenario_id,
            tracking_uri=_tracking_uri(),
        ).reload()
        candidate = _runtime_from_loaded_model(
            loaded
        )
    except (
        FileNotFoundError,
        ImportError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        raise HTTPException(
            status_code=503,
            detail=f"Model reload failed: {error}",
        ) from error

    _swap_runtime(candidate)

    return {
        "accepted": True,
        "reload_status": "reloaded",
        "previous": previous.to_dict(),
        "active": candidate.to_dict(),
        "source_of_truth": "mlflow_registry",
    }


@app.post("/predict")
def predict(
    request: InferenceServiceRequest,
) -> dict[str, str | float | None]:
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
        result = _run_real_inference(
            inference_request
        )
    except FileNotFoundError as error:
        detail = str(error)
        if detail.startswith(
            "Input image not found"
        ):
            raise HTTPException(
                status_code=404,
                detail=detail,
            ) from error
        raise HTTPException(
            status_code=503,
            detail=(
                f"Inference unavailable: {detail}"
            ),
        ) from error
    except ValueError as error:
        detail = str(error)
        if (
            "Input image checksum mismatch"
            in detail
        ):
            raise HTTPException(
                status_code=422,
                detail=detail,
            ) from error
        raise HTTPException(
            status_code=503,
            detail=(
                f"Inference unavailable: {detail}"
            ),
        ) from error
    except (
        ImportError,
        RuntimeError,
        OSError,
    ) as error:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Inference unavailable: {error}"
            ),
        ) from error

    return result.to_dict()


__all__ = [
    "ActiveInferenceRuntime",
    "InferenceServiceRequest",
    "ReloadInferenceRequest",
    "app",
    "health",
    "lifespan",
    "metrics",
    "model_version",
    "predict",
    "reload_model",
]
