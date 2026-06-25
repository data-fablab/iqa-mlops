"""Small lifecycle trigger rules used by monitoring and Airflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass

NATURAL_REPLAY_SCENARIO_ID = "production_replay_natural"
NATURAL_TRAIN_REPLAY_SCENARIO_ID = "production_replay_natural_train_v004"
NATURAL_REPLAY_SCENARIO_IDS = frozenset(
    {
        NATURAL_REPLAY_SCENARIO_ID,
        NATURAL_TRAIN_REPLAY_SCENARIO_ID,
    }
)
DRIFT_REPLAY_SCENARIO_ID = "drift_domain_extension"
NATURAL_CONFORMING_VALIDATED_TRIGGER_COUNT = 50
FEATURE_AE_V002_DATASET_VERSION = "feature_ae_good_mvp_v001"
FEATURE_AE_V003_DATASET_VERSION = "feature_ae_good_mvp_v001"


@dataclass(frozen=True)
class LifecycleSignal:
    scenario_id: str
    conforming_validated_count: int
    drift_confirmed: bool
    roi_fail_rate: float = 0.0

    def to_dict(self) -> dict[str, str | int | float | bool]:
        return asdict(self)


@dataclass(frozen=True)
class LifecycleDecision:
    scenario_id: str
    trigger_lifecycle: bool
    trigger_reason: str
    candidate_dataset_version: str | None
    conforming_validated_count: int
    drift_confirmed: bool

    def to_dict(self) -> dict[str, str | int | bool | None]:
        return asdict(self)


def evaluate_lifecycle_signal(
    signal: LifecycleSignal,
    *,
    min_natural_conforming: int = NATURAL_CONFORMING_VALIDATED_TRIGGER_COUNT,
) -> LifecycleDecision:
    """Evaluate data-event lifecycle rules without launching training."""

    if signal.scenario_id in NATURAL_REPLAY_SCENARIO_IDS:
        count = signal.conforming_validated_count
        triggered = count >= int(min_natural_conforming)
        return LifecycleDecision(
            scenario_id=signal.scenario_id,
            trigger_lifecycle=triggered,
            trigger_reason=(
                "natural_50_oracle_conformes"
                if triggered
                else "natural_waiting_for_50_oracle_conformes"
            ),
            candidate_dataset_version=FEATURE_AE_V002_DATASET_VERSION if triggered else None,
            conforming_validated_count=count,
            drift_confirmed=signal.drift_confirmed,
        )

    if signal.scenario_id == DRIFT_REPLAY_SCENARIO_ID:
        return LifecycleDecision(
            scenario_id=signal.scenario_id,
            trigger_lifecycle=signal.drift_confirmed,
            trigger_reason="drift_confirmed" if signal.drift_confirmed else "drift_not_confirmed",
            candidate_dataset_version=FEATURE_AE_V003_DATASET_VERSION if signal.drift_confirmed else None,
            conforming_validated_count=signal.conforming_validated_count,
            drift_confirmed=signal.drift_confirmed,
        )

    return LifecycleDecision(
        scenario_id=signal.scenario_id,
        trigger_lifecycle=False,
        trigger_reason="unsupported_scenario",
        candidate_dataset_version=None,
        conforming_validated_count=signal.conforming_validated_count,
        drift_confirmed=signal.drift_confirmed,
    )


def should_trigger_lifecycle(signal: LifecycleSignal, *, min_conforming: int = 50, max_roi_fail_rate: float = 0.02) -> bool:
    _ = max_roi_fail_rate
    return evaluate_lifecycle_signal(signal, min_natural_conforming=min_conforming).trigger_lifecycle


__all__ = [
    "DRIFT_REPLAY_SCENARIO_ID",
    "FEATURE_AE_V002_DATASET_VERSION",
    "FEATURE_AE_V003_DATASET_VERSION",
    "LifecycleDecision",
    "LifecycleSignal",
    "NATURAL_CONFORMING_VALIDATED_TRIGGER_COUNT",
    "NATURAL_REPLAY_SCENARIO_ID",
    "NATURAL_REPLAY_SCENARIO_IDS",
    "NATURAL_TRAIN_REPLAY_SCENARIO_ID",
    "evaluate_lifecycle_signal",
    "should_trigger_lifecycle",
]
