"""Deterministic, GPU-free domain-drift scorer for the controlled drift demo.

The retained Feature-AE is trained on the ``Casting_class1`` baseline. The
controlled drift replay (``drift_domain_extension``) replays that baseline first
(phase ``baseline_domain_class1``) then introduces ``Casting_class2`` and
``Casting_class3`` pieces (phases ``domain_extension_class2`` / ``_class3``). The
autoencoder reconstructs these out-of-distribution domains poorly, so the anomaly
score rises and the decision escalates ``Vert -> Orange -> Rouge``.

This module reproduces that escalation deterministically from
``(scenario_id, source_class)``, **without running torch or taking the single-GPU
lock**, so the chemin B proxy (anomaly rate on ``iqa_prediction_total``) carries a
faithful domain-drift signal. Outside the drift replay (natural production, demo
scenarios) the deployed model already covers the distribution, so the score stays
at the baseline and the decision is ``Vert``.

The chemin A (fidelity) target replaces this synthetic score with the real
Feature-AE ``reconstruction_p95`` (proposition §10); the score bands here mirror
``configs/monitoring_thresholds.yaml`` so the two paths stay aligned.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from iqa.monitoring.lifecycle import DRIFT_REPLAY_SCENARIO_ID

# The class the retained Feature-AE was trained on (drift manifest phase
# ``baseline_domain_class1``). Pieces of this class are in-distribution.
MODEL_BASELINE_SOURCE_CLASS = "Casting_class1"

# Decision bands mirror configs/monitoring_thresholds.yaml reconstruction_p95:
# below warning -> Vert, [warning, critical) -> Orange, >= critical -> Rouge.
RECONSTRUCTION_WARNING = 0.75
RECONSTRUCTION_CRITICAL = 0.90

# Synthetic reconstruction-error score by domain distance from the baseline.
BASELINE_SCORE = 0.10
DOMAIN_EXTENSION_SCORES: dict[str, float] = {
    "Casting_class2": 0.82,  # near domain extension -> Orange
    "Casting_class3": 0.96,  # far domain extension  -> Rouge
}

# Representative score of each severity band, derived from the decision thresholds
# (RECONSTRUCTION_WARNING / RECONSTRUCTION_CRITICAL) -- NOT from a per-class
# constant. A band is a severity level any uncovered class passes through, so
# both Casting_class2 and Casting_class3 use the SAME Orange/Rouge band scores.
ORANGE_BAND_SCORE = 0.82  # in [RECONSTRUCTION_WARNING, RECONSTRUCTION_CRITICAL) -> Orange
RED_BAND_SCORE = 0.96     # >= RECONSTRUCTION_CRITICAL                            -> Rouge

# Live drift state file (demo orchestration). With only fake/random checkpoints we
# cannot read a real reconstruction error that would *recover* after a retrain, so
# the orchestrator drives a per-class band here and the retrain marks a class
# "covered" once iqa_lifecycle has (re)trained on it. The scorer reads this file on
# every call so the decision tracks the model's evolving coverage:
#   {"Casting_class2": "orange" | "red" | "covered", ...}
# Bands map to the synthetic reconstruction score so notif/panel/sensor stay aligned.
DRIFT_STATE_ENV = "IQA_DRIFT_STATE_FILE"
_STATE_BANDS = {
    "covered": BASELINE_SCORE,  # 0.10 -> Vert
    "vert": BASELINE_SCORE,
    "green": BASELINE_SCORE,
    "orange": ORANGE_BAND_SCORE,  # any uncovered class can be Orange
    "red": RED_BAND_SCORE,        # any uncovered class can be Rouge
    "rouge": RED_BAND_SCORE,
}


def _live_drift_state() -> dict[str, str]:
    """Per-class band overrides from the orchestrator state file (best-effort)."""
    path = os.environ.get(DRIFT_STATE_ENV)
    if not path:
        return {}
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    classes = raw.get("classes", raw) if isinstance(raw, dict) else {}
    return {str(k): str(v).lower() for k, v in classes.items()} if isinstance(classes, dict) else {}


def domain_anomaly_score(scenario_id: str, source_class: str | None) -> float:
    """Synthetic Feature-AE reconstruction score for an event.

    Only the controlled drift replay escalates: in any other scenario the
    deployed model covers the distribution, so the score is the baseline. Within
    the drift replay, a live state file (if configured) overrides the band per
    source class so the decision recovers to Vert once the class is "covered".
    """
    if scenario_id != DRIFT_REPLAY_SCENARIO_ID:
        return BASELINE_SCORE
    state = _live_drift_state()
    band = state.get(source_class or "")
    if band in _STATE_BANDS:
        return _STATE_BANDS[band]
    return DOMAIN_EXTENSION_SCORES.get(source_class or "", BASELINE_SCORE)


def decision_for_score(score: float) -> str:
    """Map a reconstruction score to a {Vert, Orange, Rouge} decision."""
    if score >= RECONSTRUCTION_CRITICAL:
        return "Rouge"
    if score >= RECONSTRUCTION_WARNING:
        return "Orange"
    return "Vert"


__all__ = [
    "BASELINE_SCORE",
    "DOMAIN_EXTENSION_SCORES",
    "MODEL_BASELINE_SOURCE_CLASS",
    "RECONSTRUCTION_CRITICAL",
    "RECONSTRUCTION_WARNING",
    "decision_for_score",
    "domain_anomaly_score",
]
