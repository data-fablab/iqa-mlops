"""Contract: iqa_prediction_total is segregated by scenario_id.

The proxy drift signal filters ``scenario_id=~"drift.*"`` on
``iqa_prediction_total``; without the label the drift regime is
indistinguishable from natural traffic at the Prometheus level (issue 02).
"""

from __future__ import annotations

import pytest
from prometheus_client.parser import text_string_to_metric_families

from iqa.api.main import (
    PREDICTION_DECISION_COUNTS,
    PREDICTION_STORE,
    metrics,
    predict,
)
from iqa.api.schemas import PredictRequest


@pytest.fixture(autouse=True)
def _reset_state() -> None:
    PREDICTION_DECISION_COUNTS.clear()
    PREDICTION_STORE.clear()
    yield
    PREDICTION_DECISION_COUNTS.clear()
    PREDICTION_STORE.clear()


def _predict(scenario_id: str, suffix: str) -> None:
    predict(
        PredictRequest(
            piece_event_id=f"piece_{suffix}",
            scenario_id=scenario_id,
            image_uri=f"s3://iqa/raw/piece_{suffix}.png",
            sha256=(suffix[0] * 64)[:64],
            lot_id=f"lot_{suffix}",
            source_class="Casting_class1",
            dataset_version="casting_v010",
        )
    )


def _prediction_samples() -> list:
    body = metrics()
    samples = [
        sample
        for family in text_string_to_metric_families(body)
        for sample in family.samples
        if sample.name == "iqa_prediction_total"
    ]
    if not samples:
        raise AssertionError("iqa_prediction_total not exposed")
    return samples


def test_prediction_total_carries_decision_and_scenario_labels() -> None:
    _predict("drift_domain_extension", "drift1")

    samples = _prediction_samples()
    assert samples, "no iqa_prediction_total samples emitted"
    for sample in samples:
        assert "decision" in sample.labels
        assert "scenario_id" in sample.labels


def test_drift_and_natural_traffic_are_distinct_series() -> None:
    _predict("drift_domain_extension", "drift1")
    _predict("drift_domain_extension", "drift2")
    _predict("production_replay_natural", "nat1")

    by_scenario: dict[str, float] = {}
    for sample in _prediction_samples():
        scenario = sample.labels["scenario_id"]
        by_scenario[scenario] = by_scenario.get(scenario, 0.0) + sample.value

    assert by_scenario["drift_domain_extension"] == 2
    assert by_scenario["production_replay_natural"] == 1


def test_metrics_output_still_parses() -> None:
    _predict("drift_domain_extension", "drift1")
    # Raises on duplicate names / malformed HELP/TYPE.
    families = list(text_string_to_metric_families(metrics()))
    assert families
