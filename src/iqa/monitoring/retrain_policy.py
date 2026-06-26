"""Multi-signal retrain policy evaluator (Issues 15-18, ADR 0010 amendment).

Pure function ``evaluate_retrain_policy`` unifies three independent trigger
families — accumulation, metric floor, PatchCore drift — and adds anti-loop
guards plus HITL escalation. The sensor-DAG calls this function periodically
in pull mode; the function itself does no I/O.

Trigger priority for labelling when multiple fire: drift > metric_floor > accumulation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from iqa.monitoring.lifecycle import (
    NATURAL_CONFORMING_VALIDATED_TRIGGER_COUNT,
    evaluate_lifecycle_signal,
    LifecycleSignal,
)

TRIGGER_PRIORITY = ("drift", "metric_floor", "accumulation")

DEFAULT_METRIC_FLOOR = {
    "pixel_aupimo_1e-5_1e-3": 0.15,
    "pixel_ap": 0.20,
}

DEFAULT_MAX_GATE_FAILURES = 2
DEFAULT_COOLDOWN_SECONDS = 900
DEFAULT_OOD_RATIO_THRESHOLD = 0.5


@dataclass(frozen=True)
class RetrainPolicySignal:
    """All inputs the policy evaluator needs — assembled by the sensor in pull."""

    conforming_validated_count: int = 0
    drift_confirmed: bool = False
    drift_triggering_class: str | None = None
    drift_ood_ratio: float = 0.0
    prod_metrics: dict[str, float] = field(default_factory=dict)
    last_trigger_inputs: dict[str, Any] | None = None
    gate_failure_count: int = 0
    seconds_since_last_attempt: float | None = None
    active_lifecycle_run: bool = False


@dataclass(frozen=True)
class RetrainPolicyDecision:
    """Output of the policy evaluator — consumed by the sensor to trigger or not."""

    trigger: bool
    trigger_reasons: list[str]
    primary_reason: str
    candidate_dataset_version: str | None
    retrain_scope: str
    triggering_class: str | None
    all_fired_reasons: list[str]
    blocked_reason: str | None = None
    hitl_escalation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _check_accumulation(
    signal: RetrainPolicySignal,
    *,
    min_conforming: int = NATURAL_CONFORMING_VALIDATED_TRIGGER_COUNT,
) -> tuple[bool, str | None]:
    if signal.conforming_validated_count >= min_conforming:
        return True, "accumulation"
    return False, None


def _check_metric_floor(
    signal: RetrainPolicySignal,
    *,
    floor_targets: dict[str, float] | None = None,
) -> tuple[bool, str | None]:
    targets = floor_targets or DEFAULT_METRIC_FLOOR
    if not signal.prod_metrics:
        return False, None
    for metric_key, target_value in targets.items():
        prod_value = signal.prod_metrics.get(metric_key)
        if prod_value is not None and prod_value < target_value:
            return True, "metric_floor"
    return False, None


def _check_drift(
    signal: RetrainPolicySignal,
    *,
    ood_ratio_threshold: float = DEFAULT_OOD_RATIO_THRESHOLD,
) -> tuple[bool, str | None]:
    if signal.drift_confirmed:
        return True, "drift"
    if signal.drift_ood_ratio > ood_ratio_threshold:
        return True, "drift"
    return False, None


def _inputs_changed(signal: RetrainPolicySignal, fired_reasons: list[str]) -> bool:
    """Re-trigger only if an input changed since the last attempt (Issue 18)."""
    prev = signal.last_trigger_inputs
    if prev is None:
        return True
    if "accumulation" in fired_reasons:
        if signal.conforming_validated_count != prev.get("conforming_validated_count"):
            return True
    if "drift" in fired_reasons:
        if signal.drift_triggering_class != prev.get("drift_triggering_class"):
            return True
    if "metric_floor" in fired_reasons:
        if signal.prod_metrics != prev.get("prod_metrics"):
            return True
    return False


def _resolve_scope(fired_reasons: list[str], triggering_class: str | None) -> str:
    """Compute the retrain scope — union when multiple triggers fire (Issue 18).

    drift scope (incremental coverage) subsumes metric_floor scope (all good),
    which subsumes accumulation scope (bootstrap).
    """
    if "drift" in fired_reasons:
        return "incremental_coverage"
    if "metric_floor" in fired_reasons:
        return "full_domain_good"
    return "bootstrap"


def _resolve_dataset_version(
    fired_reasons: list[str],
    triggering_class: str | None,
) -> str | None:
    if "drift" in fired_reasons and triggering_class:
        return f"feature_ae_drift_{triggering_class}"
    if "metric_floor" in fired_reasons:
        return "feature_ae_full_retrain"
    if "accumulation" in fired_reasons:
        return "feature_ae_good_v002"
    return None


def evaluate_retrain_policy(
    signal: RetrainPolicySignal,
    *,
    min_conforming: int = NATURAL_CONFORMING_VALIDATED_TRIGGER_COUNT,
    floor_targets: dict[str, float] | None = None,
    ood_ratio_threshold: float = DEFAULT_OOD_RATIO_THRESHOLD,
    max_gate_failures: int = DEFAULT_MAX_GATE_FAILURES,
    cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
) -> RetrainPolicyDecision:
    """Evaluate the multi-signal retrain policy (pure, no I/O).

    Returns a decision with trigger=True only if at least one signal fires
    AND none of the anti-loop guards block it.
    """
    fired: list[str] = []

    ok_acc, reason_acc = _check_accumulation(signal, min_conforming=min_conforming)
    if ok_acc and reason_acc:
        fired.append(reason_acc)

    ok_floor, reason_floor = _check_metric_floor(signal, floor_targets=floor_targets)
    if ok_floor and reason_floor:
        fired.append(reason_floor)

    ok_drift, reason_drift = _check_drift(signal, ood_ratio_threshold=ood_ratio_threshold)
    if ok_drift and reason_drift:
        fired.append(reason_drift)

    if not fired:
        return RetrainPolicyDecision(
            trigger=False,
            trigger_reasons=[],
            primary_reason="no_trigger",
            candidate_dataset_version=None,
            retrain_scope="none",
            triggering_class=None,
            all_fired_reasons=[],
        )

    primary = next(r for r in TRIGGER_PRIORITY if r in fired)
    triggering_class = signal.drift_triggering_class if "drift" in fired else None
    scope = _resolve_scope(fired, triggering_class)
    dataset_version = _resolve_dataset_version(fired, triggering_class)

    # --- Anti-loop guards (Issue 18) ---

    if signal.active_lifecycle_run:
        return RetrainPolicyDecision(
            trigger=False,
            trigger_reasons=fired,
            primary_reason=primary,
            candidate_dataset_version=dataset_version,
            retrain_scope=scope,
            triggering_class=triggering_class,
            all_fired_reasons=fired,
            blocked_reason="lifecycle_run_in_flight",
        )

    if signal.gate_failure_count >= max_gate_failures:
        return RetrainPolicyDecision(
            trigger=False,
            trigger_reasons=fired,
            primary_reason=primary,
            candidate_dataset_version=dataset_version,
            retrain_scope=scope,
            triggering_class=triggering_class,
            all_fired_reasons=fired,
            blocked_reason="hitl_escalation_max_failures",
            hitl_escalation=True,
        )

    if (
        signal.seconds_since_last_attempt is not None
        and signal.seconds_since_last_attempt < cooldown_seconds
    ):
        return RetrainPolicyDecision(
            trigger=False,
            trigger_reasons=fired,
            primary_reason=primary,
            candidate_dataset_version=dataset_version,
            retrain_scope=scope,
            triggering_class=triggering_class,
            all_fired_reasons=fired,
            blocked_reason="cooldown_active",
        )

    if not _inputs_changed(signal, fired):
        return RetrainPolicyDecision(
            trigger=False,
            trigger_reasons=fired,
            primary_reason=primary,
            candidate_dataset_version=dataset_version,
            retrain_scope=scope,
            triggering_class=triggering_class,
            all_fired_reasons=fired,
            blocked_reason="inputs_unchanged",
        )

    return RetrainPolicyDecision(
        trigger=True,
        trigger_reasons=fired,
        primary_reason=primary,
        candidate_dataset_version=dataset_version,
        retrain_scope=scope,
        triggering_class=triggering_class,
        all_fired_reasons=fired,
    )


def retrain_policy_parity_with_lifecycle(
    signal: RetrainPolicySignal,
    *,
    min_conforming: int = NATURAL_CONFORMING_VALIDATED_TRIGGER_COUNT,
) -> bool:
    """Non-regression check: accumulation trigger agrees with the old path."""
    legacy = evaluate_lifecycle_signal(
        LifecycleSignal(
            scenario_id="production_replay_natural",
            conforming_validated_count=signal.conforming_validated_count,
            drift_confirmed=signal.drift_confirmed,
        ),
        min_natural_conforming=min_conforming,
    )
    new = evaluate_retrain_policy(signal, min_conforming=min_conforming)
    accumulation_fired = "accumulation" in new.all_fired_reasons
    return legacy.trigger_lifecycle == accumulation_fired or (
        legacy.trigger_lifecycle and new.trigger
    )


__all__ = [
    "DEFAULT_COOLDOWN_SECONDS",
    "DEFAULT_MAX_GATE_FAILURES",
    "DEFAULT_METRIC_FLOOR",
    "DEFAULT_OOD_RATIO_THRESHOLD",
    "TRIGGER_PRIORITY",
    "RetrainPolicyDecision",
    "RetrainPolicySignal",
    "evaluate_retrain_policy",
    "retrain_policy_parity_with_lifecycle",
]
