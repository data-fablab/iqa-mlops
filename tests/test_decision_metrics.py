"""Tests for decision metrics (recall, orange_rate, latency) used by the recall gate."""

from __future__ import annotations

from iqa.promotion.gates import evaluate_recall_gate
from iqa.training.feature_ae_evaluation import compute_decision_metrics


class TestComputeDecisionMetrics:
    """Decision metrics derived from image labels, scores and thresholds."""

    def test_false_negative_drops_recall_below_one(self) -> None:
        """A defective image scored below threshold_orange is a false negative."""
        # Two defects; the second scores green (missed) -> recall = 0.5.
        labels = [True, True, False]
        scores = [0.09, 0.001, 0.001]

        metrics = compute_decision_metrics(
            labels, scores, threshold_orange=0.02, threshold_red=0.05
        )

        assert metrics["recall"] < 1.0
        assert metrics["false_negatives"] == 1

    def test_all_defects_detected_gives_perfect_recall(self) -> None:
        """Every defect flagged (orange or red) gives recall 1.0, no false negatives."""
        labels = [True, True, False]
        scores = [0.03, 0.09, 0.001]

        metrics = compute_decision_metrics(
            labels, scores, threshold_orange=0.02, threshold_red=0.05
        )

        assert metrics["recall"] == 1.0
        assert metrics["false_negatives"] == 0

    def test_false_negative_blocks_recall_gate(self) -> None:
        """A missed defect drops recall below 1.0 and blocks promotion."""
        labels = [True, True]
        scores = [0.09, 0.001]  # second defect missed

        metrics = compute_decision_metrics(
            labels, scores, threshold_orange=0.02, threshold_red=0.05
        )
        gate = evaluate_recall_gate(metrics["recall"], threshold=1.0)

        assert gate["passed"] is False

    def test_full_recall_passes_recall_gate(self) -> None:
        """All defects caught -> recall gate passes."""
        metrics = compute_decision_metrics(
            [True, True], [0.09, 0.03], threshold_orange=0.02, threshold_red=0.05
        )
        gate = evaluate_recall_gate(metrics["recall"], threshold=1.0)

        assert gate["passed"] is True

    def test_orange_rate_counts_only_scores_between_thresholds(self) -> None:
        """Orange rate is the fraction of images scored in [orange, red)."""
        # green, orange, orange, red -> 2 of 4 are orange.
        scores = [0.001, 0.03, 0.049, 0.09]
        labels = [False, False, False, True]

        metrics = compute_decision_metrics(
            labels, scores, threshold_orange=0.02, threshold_red=0.05
        )

        assert metrics["orange_rate"] == 0.5

    def test_latency_ms_is_p95_of_observed_latencies(self) -> None:
        """Latency reported is the p95 of per-image inference latencies."""
        latencies = [10.0] * 19 + [100.0]

        metrics = compute_decision_metrics(
            [True], [0.09], threshold_orange=0.02, threshold_red=0.05,
            latencies_ms=latencies,
        )

        # p95 of this distribution sits between the bulk (10) and the tail (100).
        assert 10.0 < metrics["latency_ms"] <= 100.0

    def test_no_defects_gives_perfect_recall(self) -> None:
        """With no defective images recall is 1.0 (no division by zero)."""
        metrics = compute_decision_metrics(
            [False, False], [0.001, 0.5], threshold_orange=0.02, threshold_red=0.05
        )

        assert metrics["recall"] == 1.0
        assert metrics["false_negatives"] == 0
