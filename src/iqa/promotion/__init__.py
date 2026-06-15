"""Promotion gates and decision logic."""

from __future__ import annotations

from iqa.promotion.defect_coverage import (
    check_defect_coverage_gate,
    compute_defect_coverage,
    compute_defect_coverage_from_manifest,
)
from iqa.promotion.gates import (
    evaluate_ap_regression_gate,
    evaluate_latency_gate,
    evaluate_orange_rate_gate,
    evaluate_promotion_gates,
    evaluate_recall_gate,
)
from iqa.promotion.promotion import (
    evaluate_gates_for_promotion,
    promote_model_with_gates,
    resolve_model_artifacts,
    transition_model_stage,
)
from iqa.promotion.rollback import (
    get_previous_prod,
    rollback_model,
    save_previous_prod_before_promotion,
)

__all__ = [
    "compute_defect_coverage",
    "check_defect_coverage_gate",
    "compute_defect_coverage_from_manifest",
    "evaluate_recall_gate",
    "evaluate_ap_regression_gate",
    "evaluate_orange_rate_gate",
    "evaluate_latency_gate",
    "evaluate_promotion_gates",
    "evaluate_gates_for_promotion",
    "transition_model_stage",
    "resolve_model_artifacts",
    "promote_model_with_gates",
    "save_previous_prod_before_promotion",
    "get_previous_prod",
    "rollback_model",
]
