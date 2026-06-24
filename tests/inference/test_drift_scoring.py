"""Contract tests for the controlled domain-drift scorer (chemin B proxy signal).

The retained Feature-AE is trained on the Casting_class1 baseline. The drift
replay introduces Casting_class2 / Casting_class3 as out-of-distribution domains,
which must escalate the decision Vert -> Orange -> Rouge so the anomaly-rate proxy
carries a faithful drift signal. Any non-drift scenario must stay in-distribution.
"""

from __future__ import annotations

import pytest

from iqa.inference.contracts import InferenceRequest, placeholder_inference
from iqa.inference.drift_scoring import (
    MODEL_BASELINE_SOURCE_CLASS,
    decision_for_score,
    domain_anomaly_score,
)
from iqa.monitoring.lifecycle import DRIFT_REPLAY_SCENARIO_ID, NATURAL_REPLAY_SCENARIO_ID


def _decision(scenario_id: str, source_class: str | None) -> str:
    return placeholder_inference(
        InferenceRequest(
            piece_event_id="piece-1",
            scenario_id=scenario_id,
            image_uri="memory://x",
            source_class=source_class,
        )
    ).decision


class TestDomainDriftEscalation:
    def test_drift_baseline_class_is_in_distribution(self) -> None:
        assert _decision(DRIFT_REPLAY_SCENARIO_ID, MODEL_BASELINE_SOURCE_CLASS) == "Vert"

    def test_drift_extension_class2_is_orange(self) -> None:
        assert _decision(DRIFT_REPLAY_SCENARIO_ID, "Casting_class2") == "Orange"

    def test_drift_extension_class3_is_rouge(self) -> None:
        assert _decision(DRIFT_REPLAY_SCENARIO_ID, "Casting_class3") == "Rouge"

    def test_natural_regime_stays_vert_even_for_extension_classes(self) -> None:
        # The deployed production model covers the production distribution.
        assert _decision(NATURAL_REPLAY_SCENARIO_ID, "Casting_class3") == "Vert"

    def test_unknown_scenario_stays_vert(self) -> None:
        assert _decision("demo", None) == "Vert"

    def test_drift_unknown_class_defaults_to_baseline(self) -> None:
        assert _decision(DRIFT_REPLAY_SCENARIO_ID, "Casting_classX") == "Vert"


class TestScoreBands:
    def test_score_monotonic_with_domain_distance(self) -> None:
        baseline = domain_anomaly_score(DRIFT_REPLAY_SCENARIO_ID, "Casting_class1")
        class2 = domain_anomaly_score(DRIFT_REPLAY_SCENARIO_ID, "Casting_class2")
        class3 = domain_anomaly_score(DRIFT_REPLAY_SCENARIO_ID, "Casting_class3")
        assert baseline < class2 < class3

    @pytest.mark.parametrize(
        "score,expected",
        [(0.0, "Vert"), (0.74, "Vert"), (0.75, "Orange"), (0.89, "Orange"), (0.90, "Rouge"), (1.0, "Rouge")],
    )
    def test_decision_for_score_bands(self, score: float, expected: str) -> None:
        assert decision_for_score(score) == expected
