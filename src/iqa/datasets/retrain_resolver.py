"""Retrain sample resolver — static implementation A (Issue 9, ADR 0010 decision 5).

A single seam decouples the *source* of retrain samples from the downstream
pipeline (build_candidate_dataset → train → eval → gates → promote → reload).

Implementation A (now): filters the static drift plan CSV to produce incremental
coverage — class1 baseline + all classes seen up to and including the triggering
class. Only ``good`` (non-defective) samples are selected because the Feature-AE
trains on nominal images.

Implementation C (later): queries a feedback store. Swaps by flag, downstream
unchanged.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_PLAN_PATH = "data/metadata/casting_flux_replay_plan_drift.csv"

PHASE_ORDER = (
    "baseline_domain_class1",
    "domain_extension_class2",
    "domain_extension_class3",
)

CLASS_TO_PHASE = {
    "Casting_class1": "baseline_domain_class1",
    "Casting_class2": "domain_extension_class2",
    "Casting_class3": "domain_extension_class3",
}


@dataclass(frozen=True)
class RetrainTrigger:
    """Context passed by the sensor to scope the retrain dataset."""

    scenario_id: str
    triggering_class: str
    triggered_at: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "RetrainTrigger":
        return cls(
            scenario_id=str(payload["scenario_id"]),
            triggering_class=str(payload["triggering_class"]),
            triggered_at=payload.get("triggered_at"),
        )


@dataclass(frozen=True)
class RetrainSample:
    """One image eligible for the retrain dataset."""

    piece_event_id: str
    source_class: str
    image_uri: str
    label: str
    scenario_phase: str


def _phases_up_to(triggering_class: str) -> set[str]:
    """Return all phases up to and including the triggering class's phase."""
    phase = CLASS_TO_PHASE.get(triggering_class)
    if phase is None:
        return set(PHASE_ORDER)
    cutoff = PHASE_ORDER.index(phase)
    return set(PHASE_ORDER[: cutoff + 1])


def resolve_retrain_samples(
    trigger: RetrainTrigger,
    *,
    plan_path: str | Path | None = None,
    image_root: str | Path | None = None,
) -> list[RetrainSample]:
    """Static resolver A: filter the drift plan for incremental coverage.

    Returns only ``good`` (non-defective) samples from all phases up to and
    including the triggering class. The downstream ``build_candidate_dataset``
    consumes the result without knowing which resolver produced it.
    """
    plan = Path(plan_path or DEFAULT_PLAN_PATH)
    root = Path(image_root) if image_root else None
    eligible_phases = _phases_up_to(trigger.triggering_class)

    samples: list[RetrainSample] = []
    with plan.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("scenario_phase") not in eligible_phases:
                continue
            if row.get("label") != "good":
                continue
            relative_paths = row.get("relative_paths", "")
            for rel_path in relative_paths.split("|"):
                rel_path = rel_path.strip()
                if not rel_path:
                    continue
                uri = str(root / rel_path) if root else rel_path
                samples.append(
                    RetrainSample(
                        piece_event_id=row.get("piece_event_id", ""),
                        source_class=row.get("source_class", ""),
                        image_uri=uri,
                        label="good",
                        scenario_phase=row.get("scenario_phase", ""),
                    )
                )
    return samples


__all__ = [
    "CLASS_TO_PHASE",
    "DEFAULT_PLAN_PATH",
    "PHASE_ORDER",
    "RetrainSample",
    "RetrainTrigger",
    "resolve_retrain_samples",
]
