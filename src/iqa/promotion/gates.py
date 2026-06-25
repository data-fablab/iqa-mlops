"""Promotion gates for model evaluation and decision making."""

from __future__ import annotations

import math
from typing import Any

# Business-metric priority order and AUPIMO key, single-sourced from the metric
# contract. Importing this stays torch-free (model_metrics only pulls ``typing``),
# so ``iqa.promotion.gates`` keeps running on the data image (ADR 0008, issue 10).
from iqa.monitoring.model_metrics import AUPIMO_KEY, MODEL_QUALITY_METRIC_KEYS

# Default max allowed drop vs prod baseline for a business metric (ADR 0010 §6).
DEFAULT_QUALITY_MAX_REGRESSION = 0.02
# Pixel-level metrics require GT masks; their absence triggers the image_ap fallback.
_PIXEL_METRIC_KEYS = (AUPIMO_KEY, "pixel_ap")
QUALITY_FALLBACK_METRIC = "image_ap"


def _finite_float(value: Any) -> float | None:
    """Coerce to a finite float, or ``None`` when missing/non-finite."""
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def evaluate_quality_regression_gates(
    candidate_metrics: dict[str, Any] | None,
    prod_metrics: dict[str, Any] | None,
    *,
    max_regressions: dict[str, float] | None = None,
    default_max_regression: float = DEFAULT_QUALITY_MAX_REGRESSION,
) -> dict[str, Any]:
    """Non-regression gate over the 4 business metrics vs the prod baseline.

    Evaluates ``pixel_aupimo_1e-5_1e-3 -> pixel_ap -> image_ap -> image_auroc``
    (priority order). A metric is only *evaluable* when both candidate and prod
    carry a finite value; pixel metrics are absent when GT masks are missing, so
    the gate falls back to ``image_ap`` (ADR 0010 §6). For each evaluable metric
    the regression is ``prod - candidate`` (higher is better) and the gate passes
    when ``regression <= max_regression``. The overall verdict passes when every
    evaluable metric passes; with no evaluable metric (no prod baseline) it is a
    vacuous pass, since a non-regression gate needs a baseline to compare against.

    Returns a dict with:
    - ``all_passed``: overall non-regression verdict.
    - ``metrics``: per-metric verdict (candidate, prod, regression, max_regression).
    - ``evaluated_metrics`` / ``skipped_metrics``: metric keys, in priority order.
    - ``decisive_metric``: the highest-priority evaluable metric (reference choice).
    - ``fallback_to_image_ap``: True when no pixel metric was evaluable.
    """
    candidate_metrics = candidate_metrics or {}
    prod_metrics = prod_metrics or {}
    max_regressions = max_regressions or {}

    per_metric: dict[str, dict[str, Any]] = {}
    evaluated: list[str] = []
    skipped: list[str] = []
    for key in MODEL_QUALITY_METRIC_KEYS:
        candidate_value = _finite_float(candidate_metrics.get(key))
        prod_value = _finite_float(prod_metrics.get(key))
        if candidate_value is None or prod_value is None:
            skipped.append(key)
            continue
        max_regression = float(max_regressions.get(key, default_max_regression))
        regression = prod_value - candidate_value
        per_metric[key] = {
            "passed": regression <= max_regression,
            "candidate": candidate_value,
            "prod": prod_value,
            "regression": regression,
            "max_regression": max_regression,
        }
        evaluated.append(key)

    decisive_metric = evaluated[0] if evaluated else None
    fallback_to_image_ap = bool(evaluated) and all(
        key not in evaluated for key in _PIXEL_METRIC_KEYS
    )
    return {
        "all_passed": all(result["passed"] for result in per_metric.values()),
        "metrics": per_metric,
        "evaluated_metrics": evaluated,
        "skipped_metrics": skipped,
        "decisive_metric": decisive_metric,
        "fallback_to_image_ap": fallback_to_image_ap,
    }


def evaluate_recall_gate(recall: float, threshold: float = 1.0) -> dict[str, bool | float]:
    """Evaluate recall gate (no false negatives allowed).

    Args:
        recall: Recall value (0.0 to 1.0)
        threshold: Minimum acceptable recall (default: 1.0)

    Returns:
        Dict with keys:
        - passed: bool, True if recall >= threshold
        - recall: float, the input recall value
        - threshold: float, the threshold used
    """
    return {
        "passed": recall >= threshold,
        "recall": recall,
        "threshold": threshold,
    }


def evaluate_ap_regression_gate(
    candidate_ap: float,
    prod_ap: float,
    max_regression: float = 0.02,
) -> dict[str, bool | float]:
    """Evaluate AP regression gate (regression vs production).

    Args:
        candidate_ap: Candidate model AP value
        prod_ap: Production model AP value
        max_regression: Maximum acceptable regression (default: 0.02)

    Returns:
        Dict with keys:
        - passed: bool, True if regression <= max_regression
        - candidate_ap: float
        - prod_ap: float
        - regression: float, (prod_ap - candidate_ap)
        - max_regression: float
    """
    regression = prod_ap - candidate_ap
    return {
        "passed": regression <= max_regression,
        "candidate_ap": candidate_ap,
        "prod_ap": prod_ap,
        "regression": regression,
        "max_regression": max_regression,
    }


def evaluate_orange_rate_gate(
    orange_rate: float,
    max_rate: float,
) -> dict[str, bool | float]:
    """Evaluate orange rate gate.

    Args:
        orange_rate: Orange rate value (0.0 to 1.0)
        max_rate: Maximum acceptable orange rate

    Returns:
        Dict with keys:
        - passed: bool, True if orange_rate <= max_rate
        - orange_rate: float
        - max_rate: float
    """
    return {
        "passed": orange_rate <= max_rate,
        "orange_rate": orange_rate,
        "max_rate": max_rate,
    }


def evaluate_latency_gate(
    latency_ms: float,
    max_latency_ms: float,
) -> dict[str, bool | float]:
    """Evaluate latency gate.

    Args:
        latency_ms: P95 latency in milliseconds
        max_latency_ms: Maximum acceptable latency in milliseconds

    Returns:
        Dict with keys:
        - passed: bool, True if latency_ms <= max_latency_ms
        - latency_ms: float
        - max_latency_ms: float
    """
    return {
        "passed": latency_ms <= max_latency_ms,
        "latency_ms": latency_ms,
        "max_latency_ms": max_latency_ms,
    }


def evaluate_promotion_gates(
    candidate_recall: float,
    candidate_ap: float,
    candidate_orange_rate: float,
    candidate_latency_ms: float,
    prod_ap: float | None = None,
    gates_config: dict[str, Any] | None = None,
    candidate_quality_metrics: dict[str, Any] | None = None,
    prod_quality_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate all promotion gates for a candidate model.

    Args:
        candidate_recall: Candidate model recall
        candidate_ap: Candidate model average precision
        candidate_orange_rate: Candidate model orange rate
        candidate_latency_ms: Candidate model P95 latency
        prod_ap: Production model AP (optional, legacy single-metric regression)
        gates_config: Gate thresholds from config (optional)
        candidate_quality_metrics: Candidate's 4 business metrics (optional). When
            given together with ``prod_quality_metrics`` the 4-metric non-regression
            gate (ADR 0010 §6) replaces the legacy single ``ap_regression`` gate.
        prod_quality_metrics: Prod baseline's 4 business metrics (optional).

    Returns:
        Dict with keys:
        - all_passed: bool, True if all evaluated gates pass
        - gates: dict mapping gate names to their results
        - rollback_signal: bool, True if any gate fails (triggers rollback)
    """
    if gates_config is None:
        gates_config = {}

    results = {}

    # Evaluate recall gate
    recall_threshold = gates_config.get("feature_ae", {}).get("recall_defect_min", 1.0)
    results["recall"] = evaluate_recall_gate(candidate_recall, threshold=recall_threshold)

    # Regression: prefer the 4-metric non-regression gate when the business metrics
    # of candidate and prod are both available; otherwise fall back to the legacy
    # single-metric image-AP regression gate (only if prod_ap provided).
    if candidate_quality_metrics is not None and prod_quality_metrics is not None:
        max_regressions = gates_config.get("feature_ae", {}).get(
            "quality_max_regression", {}
        )
        quality_verdict = evaluate_quality_regression_gates(
            candidate_quality_metrics,
            prod_quality_metrics,
            max_regressions=max_regressions,
        )
        results["quality_regression"] = {
            "passed": quality_verdict["all_passed"],
            "verdict": quality_verdict,
        }
    elif prod_ap is not None:
        max_regression = gates_config.get("feature_ae", {}).get("image_ap_max_regression", 0.02)
        results["ap_regression"] = evaluate_ap_regression_gate(
            candidate_ap, prod_ap, max_regression=max_regression
        )

    # Evaluate orange rate gate
    orange_rate_threshold = gates_config.get("feature_ae", {}).get("orange_rate_max", 1.0)
    results["orange_rate"] = evaluate_orange_rate_gate(
        candidate_orange_rate, orange_rate_threshold
    )

    # Evaluate latency gate
    latency_threshold = gates_config.get("feature_ae", {}).get("latency_p95_ms_max", 1000.0)
    results["latency"] = evaluate_latency_gate(candidate_latency_ms, latency_threshold)

    # Determine overall result
    all_passed = all(result["passed"] for result in results.values())

    return {
        "all_passed": all_passed,
        "gates": results,
        "rollback_signal": not all_passed,
    }


__all__ = [
    "DEFAULT_QUALITY_MAX_REGRESSION",
    "QUALITY_FALLBACK_METRIC",
    "evaluate_recall_gate",
    "evaluate_ap_regression_gate",
    "evaluate_orange_rate_gate",
    "evaluate_latency_gate",
    "evaluate_quality_regression_gates",
    "evaluate_promotion_gates",
]
