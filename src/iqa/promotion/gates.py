"""Promotion gates for model evaluation and decision making."""

from __future__ import annotations

from typing import Any


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
) -> dict[str, Any]:
    """Evaluate all promotion gates for a candidate model.

    Args:
        candidate_recall: Candidate model recall
        candidate_ap: Candidate model average precision
        candidate_orange_rate: Candidate model orange rate
        candidate_latency_ms: Candidate model P95 latency
        prod_ap: Production model AP (optional, for regression check)
        gates_config: Gate thresholds from config (optional)

    Returns:
        Dict with keys:
        - all_passed: bool, True if all evaluated gates pass
        - gates: dict mapping gate names to their results
        - rollback_signal: bool, True if any gate fails (triggers rollback)
    """
    if gates_config is None:
        gates_config = {}

    feature_ae = gates_config.get("feature_ae", {})
    results = {}

    # Evaluate recall gate
    recall_threshold = feature_ae.get("recall_defect_min", 1.0)
    results["recall"] = evaluate_recall_gate(candidate_recall, threshold=recall_threshold)

    # Evaluate AP regression gate (only if prod_ap provided)
    if prod_ap is not None:
        max_regression = feature_ae.get("image_ap_max_regression", 0.02)
        results["ap_regression"] = evaluate_ap_regression_gate(
            candidate_ap, prod_ap, max_regression=max_regression
        )

    # Evaluate orange rate gate
    orange_rate_threshold = feature_ae.get("orange_rate_max", 1.0)
    results["orange_rate"] = evaluate_orange_rate_gate(
        candidate_orange_rate, orange_rate_threshold
    )

    # Evaluate latency gate
    latency_threshold = feature_ae.get("latency_p95_ms_max", 1000.0)
    results["latency"] = evaluate_latency_gate(candidate_latency_ms, latency_threshold)

    # Determine overall result
    all_passed = all(result["passed"] for result in results.values())

    return {
        "all_passed": all_passed,
        "gates": results,
        "rollback_signal": not all_passed,
    }


__all__ = [
    "evaluate_recall_gate",
    "evaluate_ap_regression_gate",
    "evaluate_orange_rate_gate",
    "evaluate_latency_gate",
    "evaluate_promotion_gates",
]
