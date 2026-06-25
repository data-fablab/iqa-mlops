"""Promotion gates and decision logic.

Imports are lazy (PEP 562): importing a light submodule -- e.g.
``from iqa.promotion.gates import evaluate_promotion_gates`` on the torch-free
data image -- must not eagerly pull ``defect_coverage`` (-> ``iqa.datasets`` ->
``torch``) or ``promotion``/``rollback`` (-> ``mlflow``). Each public name loads
its submodule only when the attribute is actually accessed, so
``from iqa.promotion import evaluate_promotion_gates`` keeps working while
``iqa-run-gates`` runs on the data image without torch (ADR 0008, issue 10).
"""

from __future__ import annotations

import importlib

_SUBMODULE_BY_NAME = {
    "compute_defect_coverage": "defect_coverage",
    "check_defect_coverage_gate": "defect_coverage",
    "compute_defect_coverage_from_manifest": "defect_coverage",
    "evaluate_recall_gate": "gates",
    "evaluate_ap_regression_gate": "gates",
    "evaluate_orange_rate_gate": "gates",
    "evaluate_latency_gate": "gates",
    "evaluate_quality_regression_gates": "gates",
    "evaluate_promotion_gates": "gates",
    "evaluate_gates_for_promotion": "promotion",
    "transition_model_stage": "promotion",
    "resolve_model_artifacts": "promotion",
    "promote_model_with_gates": "promotion",
    "save_previous_prod_before_promotion": "rollback",
    "get_previous_prod": "rollback",
    "rollback_model": "rollback",
}

__all__ = list(_SUBMODULE_BY_NAME)


def __getattr__(name: str) -> object:
    submodule = _SUBMODULE_BY_NAME.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"{__name__}.{submodule}")
    value = getattr(module, name)
    globals()[name] = value  # cache for subsequent lookups
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
