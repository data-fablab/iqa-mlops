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

import hashlib
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import torch

logger = logging.getLogger(__name__)

# Predict-time heatmaps are emitted only for non-Vert pieces (the ones worth
# reviewing) and throttled, so a 6/s drift burst does not flood iqa-heatmaps or
# slow /predict. Both knobs are env-overridable; set the interval to 0 to disable
# throttling, or IQA_EMIT_PREDICT_HEATMAPS=0 to turn the feature off entirely.
_HEATMAP_THROTTLE_LOCK = threading.Lock()
_LAST_HEATMAP_TS = 0.0


def emit_predict_heatmaps_enabled() -> bool:
    return os.environ.get("IQA_EMIT_PREDICT_HEATMAPS", "1").strip().lower() in {"1", "true", "yes", "on"}


def _heatmap_min_interval_seconds() -> float:
    try:
        return float(os.environ.get("IQA_PREDICT_HEATMAP_MIN_INTERVAL_S", "2.0"))
    except ValueError:
        return 2.0


def _heatmap_throttle_ready() -> bool:
    """True at most once per ``min_interval`` seconds (process-wide)."""
    interval = _heatmap_min_interval_seconds()
    if interval <= 0:
        return True
    global _LAST_HEATMAP_TS
    now = time.monotonic()
    with _HEATMAP_THROTTLE_LOCK:
        if now - _LAST_HEATMAP_TS < interval:
            return False
        _LAST_HEATMAP_TS = now
        return True

from iqa.inference.contracts import Decision, InferenceRequest, InferenceResult
from iqa.inference.domain_drift import (
    DEFAULT_DETECTOR_DIR as DEFAULT_DOMAIN_DRIFT_DIR,
    PatchCoreDomainDriftDetector,
)
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


def domain_drift_enabled() -> bool:
    """Score the PatchCore domain-drift detector alongside the AE (Issue 12).

    On by default in the real path: the detector is cheap (~17 ms) and resident.
    ``IQA_DOMAIN_DRIFT=0`` turns it off (AE-only) without code changes.
    """
    return os.environ.get("IQA_DOMAIN_DRIFT", "1").strip().lower() in {"1", "true", "yes", "on"}


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
        domain_drift_dir: str | None = None,
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
        # PatchCore domain-drift detector (Issue 12): loaded lazily from the
        # registered dir (Issue 11), scored alongside the AE on each piece.
        self.domain_drift_dir = domain_drift_dir or os.environ.get(
            "IQA_DOMAIN_DRIFT_DIR", DEFAULT_DOMAIN_DRIFT_DIR
        )
        self._domain_drift: PatchCoreDomainDriftDetector | None = None
        self._domain_drift_unavailable = False

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._teacher is not None:
            return
        torch_device = torch.device(self.device)
        self._model = load_rd_feature_ae_gated(self.checkpoint_path, layers=self.layers, map_location="cpu").to(torch_device)
        self._teacher = ResNetTeacherFeatures(layers=self.layers, pretrained=True).to(torch_device)
        self._model.eval()
        self._teacher.eval()

    def reload(self, checkpoint_path: str | None = None) -> None:
        """Drop the cached model so the next score loads a fresh checkpoint.

        Also drops the cached PatchCore domain-drift detector so a rebuilt bank
        (new ``covered_classes`` after a drift retrain) is reloaded from disk on
        the next prediction. Without this the inference keeps the stale bank and a
        newly-covered class stays out-of-domain despite the refresh.
        """
        with self._lock:
            if checkpoint_path:
                self.checkpoint_path = checkpoint_path
                self.feature_ae_version = Path(checkpoint_path).parent.name or "rd_feature_ae"
            self._model = None
            self._teacher = None
            self._domain_drift = None
            self._domain_drift_unavailable = False

    def _compute_maps(self, image_path: str | Path):
        """Run the scorer once and return the full score maps (scalar + spatial map)."""
        with self._lock:
            self._ensure_loaded()
            with torch.no_grad():
                return compute_feature_ae_score_maps(
                    image_path,
                    self.checkpoint_path,
                    device=self.device,
                    layers=self.layers,
                    model=self._model,
                    teacher=self._teacher,
                )

    def score(self, image_path: str | Path) -> float:
        return float(self._compute_maps(image_path).score)

    def _ensure_domain_drift(self) -> PatchCoreDomainDriftDetector | None:
        """Lazily load the registered PatchCore detector; tolerate a missing dir."""
        if self._domain_drift is not None or self._domain_drift_unavailable:
            return self._domain_drift
        if not domain_drift_enabled():
            self._domain_drift_unavailable = True
            return None
        try:
            self._domain_drift = PatchCoreDomainDriftDetector.load(
                self.domain_drift_dir, device=self.device
            )
        except Exception:  # noqa: BLE001 - degrade to AE-only, never break /predict
            self._domain_drift_unavailable = True
            return None
        return self._domain_drift

    def domain_drift(self, image_path: str | Path) -> tuple[float | None, str | None]:
        """Score the PatchCore domain-drift detector: ``(score, regime)`` or ``(None, None)``."""
        detector = self._ensure_domain_drift()
        if detector is None:
            return None, None
        with self._lock:
            try:
                score = detector.score(image_path)
            except Exception:  # noqa: BLE001 - never break /predict on a drift hiccup
                return None, None
        return score, detector.regime(score)

    def predict(self, request: InferenceRequest) -> InferenceResult:
        image_path = resolve_image_path(request.image_uri)
        maps = self._compute_maps(image_path)
        score = float(maps.score)
        decision = self._decision(score)
        drift_score, drift_regime = self.domain_drift(image_path)
        heatmap_uri = self._maybe_emit_heatmap(image_path, maps.score_map, decision, request)
        return InferenceResult(
            piece_event_id=request.piece_event_id,
            scenario_id=request.scenario_id,
            score=score,
            decision=decision,
            heatmap_uri=heatmap_uri,
            roi_status=None,
            roi_model_version=ROI_MODEL_VERSION,
            feature_ae_version=self.feature_ae_version,
            domain_drift_score=drift_score,
            domain_regime=drift_regime,
        )

    def _maybe_emit_heatmap(self, image_path, score_map, decision: str, request: InferenceRequest) -> str | None:
        """Render + publish the anomaly heatmap for a reviewable (non-Vert) piece.

        Gated to Orange/Rouge and throttled so a drift burst neither floods
        iqa-heatmaps nor slows /predict. Best-effort: any failure returns None and
        never breaks the prediction.
        """
        if decision == "Vert" or not emit_predict_heatmaps_enabled() or not _heatmap_throttle_ready():
            return None
        try:
            from iqa.inference.feature_ae import save_feature_ae_heatmap_overlay
            from iqa.storage.visual_artifacts import VisualArtifactContext, publish_heatmap

            with tempfile.TemporaryDirectory(prefix="iqa_heatmap_") as tmp:
                out = Path(tmp) / "heatmap.png"
                save_feature_ae_heatmap_overlay(
                    image_path,
                    torch.from_numpy(score_map),
                    out,
                    threshold_orange=self.threshold_orange,
                    threshold_red=self.threshold_red,
                )
                image_id = Path(str(image_path)).stem or hashlib.sha1(str(image_path).encode()).hexdigest()[:12]
                context = VisualArtifactContext(
                    scenario_id=request.scenario_id,
                    lot_id=getattr(request, "lot_id", None) or "live",
                    piece_event_id=request.piece_event_id,
                    image_id=image_id,
                )
                return publish_heatmap(out, context)
        except Exception as exc:  # noqa: BLE001 - never break /predict on a heatmap hiccup
            logger.warning("predict heatmap emission skipped: %s", exc)
            return None

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
    "domain_drift_enabled",
    "get_scorer",
    "real_inference_enabled",
    "resolve_image_path",
]
