"""Replay scenarios exposed by the IQA API and Airflow DAGs."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ReplayScenario:
    scenario_id: str
    scenario_type: str
    purpose: str
    is_representative: bool

    def to_dict(self) -> dict[str, str | bool]:
        return asdict(self)


REPLAY_SCENARIOS = (
    ReplayScenario(
        scenario_id="production_replay_natural",
        scenario_type="production",
        purpose="Replay produit pour lots, feedback oracle et monitoring operationnel.",
        is_representative=True,
    ),
    ReplayScenario(
        scenario_id="drift_domain_extension",
        scenario_type="mlops_stress_test",
        purpose="Scenario controle pour drift, dataset candidat et promotion/rejet.",
        is_representative=False,
    ),
)


def list_replay_scenarios() -> list[dict[str, str | bool]]:
    return [scenario.to_dict() for scenario in REPLAY_SCENARIOS]


__all__ = ["REPLAY_SCENARIOS", "ReplayScenario", "list_replay_scenarios"]
