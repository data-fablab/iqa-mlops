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


def domain_anomaly_score(scenario_id: str, source_class: str | None) -> float:
    """Synthetic Feature-AE reconstruction score for an event.

    Only the controlled drift replay escalates: in any other scenario the
    deployed model covers the distribution, so the score is the baseline.
    """
    if scenario_id != DRIFT_REPLAY_SCENARIO_ID:
        return BASELINE_SCORE
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
