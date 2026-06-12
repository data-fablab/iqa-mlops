"""Small lifecycle trigger rules used by monitoring and Airflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class LifecycleSignal:
    scenario_id: str
    conforming_validated_count: int
    drift_confirmed: bool
    roi_fail_rate: float = 0.0

    def to_dict(self) -> dict[str, str | int | float | bool]:
        return asdict(self)


def should_trigger_lifecycle(signal: LifecycleSignal, *, min_conforming: int = 30, max_roi_fail_rate: float = 0.02) -> bool:
    return (
        signal.conforming_validated_count >= int(min_conforming)
        and signal.drift_confirmed
        and signal.roi_fail_rate <= float(max_roi_fail_rate)
    )


__all__ = ["LifecycleSignal", "should_trigger_lifecycle"]
