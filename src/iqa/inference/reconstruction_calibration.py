"""Calibrated reconstruction-score decision thresholds for the real scorer (Issue 6).

The ``RealFeatureAEScorer`` decision ``{Vert, Orange, Rouge}`` must come from
thresholds **calibrated on the baseline class1 score distribution** (ADR 0010 §3),
not the hardcoded ``0.02 / 0.05``. This module is the single source of truth for:

- deriving ``(orange, red)`` from a class1 score sample (high percentile + margin),
- the on-disk schema (``configs/feature_ae_reconstruction_calibration.yaml``),
- loading it at scorer init (a missing file is tolerated -> the caller falls back).

Higher reconstruction score = more anomalous: an out-of-distribution class2/class3
image scores above the class1 percentile and is flagged Orange/Rouge, while class1
stays Vert. The red threshold sits at/above the orange one with the same margin.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

# Default location of the calibration artifact (repo-relative).
DEFAULT_CALIBRATION_PATH = Path("configs/feature_ae_reconstruction_calibration.yaml")

# Default derivation knobs, aligned with the existing good-quantile calibrator
# (``scripts/calibrate_feature_ae_thresholds.py`` uses 0.95 / 0.99): Orange at the
# 95th percentile of class1, Red at the 99th, both pushed up by a small 5% margin so
# normal class1 noise stays Vert without setting the bar so high that early OOD is missed.
DEFAULT_ORANGE_PERCENTILE = 95.0
DEFAULT_RED_PERCENTILE = 99.0
DEFAULT_MARGIN = 0.05

BASELINE_PHASE = "baseline_domain_class1"


@dataclass(frozen=True)
class ReconstructionThresholds:
    """The two decision thresholds loaded from the calibration file."""

    threshold_orange: float
    threshold_red: float
    hitl_validated: bool = False


def derive_reconstruction_thresholds(
    class1_scores: Iterable[float],
    *,
    orange_percentile: float = DEFAULT_ORANGE_PERCENTILE,
    red_percentile: float = DEFAULT_RED_PERCENTILE,
    margin: float = DEFAULT_MARGIN,
) -> dict[str, Any]:
    """Derive ``(orange, red)`` from a class1 score sample (percentile + margin).

    ``threshold_orange = percentile(scores, orange_percentile) * (1 + margin)`` and
    ``threshold_red = max(percentile(scores, red_percentile) * (1 + margin), orange)``
    so red never sits below orange. Returns the thresholds plus the parameters and
    the class1 score stats, for provenance in the calibration file.
    """
    values = np.asarray(list(class1_scores), dtype=np.float64)
    if values.size == 0:
        raise ValueError("Cannot derive reconstruction thresholds from an empty class1 sample")
    if not 0.0 <= orange_percentile <= red_percentile <= 100.0:
        raise ValueError("require 0 <= orange_percentile <= red_percentile <= 100")
    if margin < 0.0:
        raise ValueError("margin must be >= 0")

    scale = 1.0 + float(margin)
    threshold_orange = float(np.percentile(values, orange_percentile)) * scale
    threshold_red = max(float(np.percentile(values, red_percentile)) * scale, threshold_orange)
    return {
        "method": "class1_percentile_plus_margin",
        "orange_percentile": float(orange_percentile),
        "red_percentile": float(red_percentile),
        "margin": float(margin),
        "threshold_orange": threshold_orange,
        "threshold_red": threshold_red,
        "class1_score_stats": {
            "count": int(values.size),
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "p99": float(np.percentile(values, 99.0)),
        },
    }


def load_reconstruction_calibration(
    path: str | Path | None = None,
) -> ReconstructionThresholds | None:
    """Load calibrated thresholds from the YAML file, or ``None`` if unavailable.

    Returns ``None`` when the file is missing or does not carry both thresholds, so
    the scorer can fall back to env vars / defaults instead of crashing. The
    ``hitl.validated`` flag is surfaced but does not gate loading (governance is the
    caller's concern).
    """
    path = Path(path or DEFAULT_CALIBRATION_PATH)
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    block = data.get("reconstruction_calibration") or {}
    thresholds = block.get("thresholds") or {}
    orange = thresholds.get("threshold_orange")
    red = thresholds.get("threshold_red")
    if orange is None or red is None:
        return None
    hitl_validated = bool((block.get("hitl") or {}).get("validated", False))
    return ReconstructionThresholds(
        threshold_orange=float(orange),
        threshold_red=float(red),
        hitl_validated=hitl_validated,
    )


__all__ = [
    "BASELINE_PHASE",
    "DEFAULT_CALIBRATION_PATH",
    "DEFAULT_MARGIN",
    "DEFAULT_ORANGE_PERCENTILE",
    "DEFAULT_RED_PERCENTILE",
    "ReconstructionThresholds",
    "derive_reconstruction_thresholds",
    "load_reconstruction_calibration",
]
