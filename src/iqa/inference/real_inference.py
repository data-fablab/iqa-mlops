"""Real Feature-AE reconstruction inference for the live drift demo.

Replaces the synthetic ``placeholder_inference`` with the genuine reverse-
distillation student-teacher scorer: the baseline checkpoint is trained on the
``Casting_class1`` distribution, so out-of-distribution ``Casting_class2`` /
``Casting_class3`` images reconstruct poorly and the anomaly score rises. The
decision ``Vert -> Orange -> Rouge`` therefore comes from the actual pixels.

The model + ResNet teacher are loaded once and cached for the process lifetime
(``RealFeatureAEScorer``); ``reload()`` drops the cache so a promoted checkpoint
is picked up after a retrain. Enabled by ``IQA_REAL_INFERENCE`` in the GPU
``iqa-inference`` service; the CPU ``iqa-api`` delegates to it over HTTP.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from urllib.parse import urlparse

import torch

from iqa.inference.contracts import Decision, InferenceRequest, InferenceResult
from iqa.inference.feature_ae import compute_feature_ae_score_maps
from iqa.inference.reconstruction_calibration import (
    DEFAULT_CALIBRATION_PATH,
    load_reconstruction_calibration,
)
from iqa.models.feature_ae import (
    DEFAULT_FEATURE_LAYERS,
    ResNetTeacherFeatures,
    load_rd_feature_ae_gated,
    normalize_feature_layers,
)

# Deployed baseline is trained on Casting_class1 only (ADR 0010 §4): class2/class3
# are then genuinely out-of-distribution.
DEFAULT_CHECKPOINT = "/opt/iqa/models/rd_feature_ae_class1_baseline/checkpoint.pt"
ROI_MODEL_VERSION = "roi_segmenter_v001_fixed"

# Last-resort thresholds, only used when neither the calibration file nor the env
# vars provide them. The calibrated file (configs/feature_ae_reconstruction_calibration.yaml)
# is the intended source of truth (ADR 0010 §3).
_FALLBACK_THRESHOLD_ORANGE = 0.02
_FALLBACK_THRESHOLD_RED = 0.05


def real_inference_enabled() -> bool:
    return os.environ.get("IQA_REAL_INFERENCE", "").strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def resolve_image_path(image_uri: str) -> Path:
    """Resolve an ``image_uri`` to a local filesystem path the scorer can read.

    Supports plain paths and ``file://`` URIs (the demo driver sends container
    paths into the bind-mounted dataset). ``s3://`` is not downloaded here.
    """
    parsed = urlparse(image_uri)
    if parsed.scheme in ("", "file"):
        return Path(parsed.path if parsed.scheme == "file" else image_uri)
    raise ValueError(f"Unsupported image_uri scheme for real inference: {image_uri!r}")


class RealFeatureAEScorer:
    """Process-lifetime cache of the Feature-AE model + teacher for scoring."""

    def __init__(
        self,
        *,
        checkpoint_path: str | None = None,
        device: str | None = None,
        threshold_orange: float | None = None,
        threshold_red: float | None = None,
        calibration_path: str | None = None,
        layers: tuple[str, ...] = DEFAULT_FEATURE_LAYERS,
    ) -> None:
        self.checkpoint_path = checkpoint_path or os.environ.get("IQA_FEATURE_AE_CHECKPOINT", DEFAULT_CHECKPOINT)
        requested = device or os.environ.get("IQA_INFERENCE_DEVICE", "cuda")
        if requested == "cuda" and not torch.cuda.is_available():
            requested = "cpu"
        self.device = requested
        # Threshold resolution (most specific first): explicit arg -> calibration
        # file (ADR 0010 §3) -> env var -> last-resort constant. The decision is no
        # longer driven by constants hardcoded in the scorer.
        self.calibration_path = calibration_path or os.environ.get(
            "IQA_FEATURE_AE_CALIBRATION", str(DEFAULT_CALIBRATION_PATH)
        )
        calibrated = load_reconstruction_calibration(self.calibration_path)
        self.threshold_source = "calibration_file" if calibrated is not None else "env_or_fallback"
        if threshold_orange is not None:
            self.threshold_orange = threshold_orange
            self.threshold_source = "explicit"
        elif calibrated is not None:
            self.threshold_orange = calibrated.threshold_orange
        else:
            self.threshold_orange = _env_float("IQA_FEATURE_AE_THRESHOLD_ORANGE", _FALLBACK_THRESHOLD_ORANGE)
        if threshold_red is not None:
            self.threshold_red = threshold_red
            self.threshold_source = "explicit"
        elif calibrated is not None:
            self.threshold_red = calibrated.threshold_red
        else:
            self.threshold_red = _env_float("IQA_FEATURE_AE_THRESHOLD_RED", _FALLBACK_THRESHOLD_RED)
        self.layers = normalize_feature_layers(layers)
        self.feature_ae_version = Path(self.checkpoint_path).parent.name or "rd_feature_ae"
        self._lock = threading.Lock()
        self._model: torch.nn.Module | None = None
        self._teacher: torch.nn.Module | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._teacher is not None:
            return
        torch_device = torch.device(self.device)
        self._model = load_rd_feature_ae_gated(self.checkpoint_path, layers=self.layers, map_location="cpu").to(torch_device)
        self._teacher = ResNetTeacherFeatures(layers=self.layers, pretrained=True).to(torch_device)
        self._model.eval()
        self._teacher.eval()

    def reload(self, checkpoint_path: str | None = None) -> None:
        """Drop the cached model so the next score loads a fresh checkpoint."""
        with self._lock:
            if checkpoint_path:
                self.checkpoint_path = checkpoint_path
                self.feature_ae_version = Path(checkpoint_path).parent.name or "rd_feature_ae"
            self._model = None
            self._teacher = None

    def score(self, image_path: str | Path) -> float:
        with self._lock:
            self._ensure_loaded()
            with torch.no_grad():
                maps = compute_feature_ae_score_maps(
                    image_path,
                    self.checkpoint_path,
                    device=self.device,
                    layers=self.layers,
                    model=self._model,
                    teacher=self._teacher,
                )
        return float(maps.score)

    def predict(self, request: InferenceRequest) -> InferenceResult:
        image_path = resolve_image_path(request.image_uri)
        score = self.score(image_path)
        decision = self._decision(score)
        return InferenceResult(
            piece_event_id=request.piece_event_id,
            scenario_id=request.scenario_id,
            score=score,
            decision=decision,
            heatmap_uri=None,
            roi_status=None,
            roi_model_version=ROI_MODEL_VERSION,
            feature_ae_version=self.feature_ae_version,
        )

    def _decision(self, score: float) -> Decision:
        if score >= self.threshold_red:
            return "Rouge"
        if score >= self.threshold_orange:
            return "Orange"
        return "Vert"


_SCORER: RealFeatureAEScorer | None = None
_SCORER_LOCK = threading.Lock()


def get_scorer() -> RealFeatureAEScorer:
    """Lazily build the process-wide scorer singleton."""
    global _SCORER
    if _SCORER is None:
        with _SCORER_LOCK:
            if _SCORER is None:
                _SCORER = RealFeatureAEScorer()
    return _SCORER


__all__ = [
    "RealFeatureAEScorer",
    "get_scorer",
    "real_inference_enabled",
    "resolve_image_path",
]
