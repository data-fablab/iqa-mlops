"""Tests for promotion gates evaluation."""

from __future__ import annotations

import pytest

from iqa.promotion.gates import (
    evaluate_recall_gate,
    evaluate_ap_regression_gate,
    evaluate_orange_rate_gate,
    evaluate_latency_gate,
    evaluate_promotion_gates,
)


class TestRecallGate:
    """Test recall gate evaluation."""

    def test_recall_gate_passes_at_perfect_recall(self) -> None:
        """Recall gate passes when recall == 1.0."""
        result = evaluate_recall_gate(recall=1.0)
        assert result["passed"] is True
        assert result["recall"] == 1.0
        assert result["threshold"] == 1.0

    def test_recall_gate_blocks_below_perfect_recall(self) -> None:
        """Recall gate blocks when recall < 1.0."""
        result = evaluate_recall_gate(recall=0.99)
        assert result["passed"] is False
        assert result["recall"] == 0.99
        assert result["threshold"] == 1.0


class TestAPRegressionGate:
    """Test AP regression gate evaluation."""

    def test_ap_regression_gate_passes_within_tolerance(self) -> None:
        """Gate passes when AP doesn't regress more than 0.02."""
        result = evaluate_ap_regression_gate(
            candidate_ap=0.93,
            prod_ap=0.95,
            max_regression=0.02,
        )
        assert result["passed"] is True
        assert result["regression"] == pytest.approx(0.02)

    def test_ap_regression_gate_passes_at_exact_threshold(self) -> None:
        """Gate passes when regression equals max_regression."""
        result = evaluate_ap_regression_gate(
            candidate_ap=0.90,
            prod_ap=0.95,
            max_regression=0.05,
        )
        assert result["passed"] is True
        assert result["regression"] == pytest.approx(0.05)

    def test_ap_regression_gate_blocks_exceeding_tolerance(self) -> None:
        """Gate blocks when AP regresses more than threshold."""
        result = evaluate_ap_regression_gate(
            candidate_ap=0.92,
            prod_ap=0.95,
            max_regression=0.02,
        )
        assert result["passed"] is False
        assert result["regression"] == pytest.approx(0.03)

    def test_ap_regression_gate_passes_when_improvement(self) -> None:
        """Gate passes when candidate AP improves over prod."""
        result = evaluate_ap_regression_gate(
            candidate_ap=0.96,
            prod_ap=0.95,
            max_regression=0.02,
        )
        assert result["passed"] is True
        assert result["regression"] == pytest.approx(-0.01)


class TestOrangeRateGate:
    """Test orange rate gate evaluation."""

    def test_orange_rate_gate_passes_below_threshold(self) -> None:
        """Gate passes when orange_rate is below threshold."""
        result = evaluate_orange_rate_gate(orange_rate=0.08, max_rate=0.10)
        assert result["passed"] is True
        assert result["orange_rate"] == 0.08

    def test_orange_rate_gate_passes_at_threshold(self) -> None:
        """Gate passes when orange_rate equals threshold."""
        result = evaluate_orange_rate_gate(orange_rate=0.10, max_rate=0.10)
        assert result["passed"] is True

    def test_orange_rate_gate_blocks_above_threshold(self) -> None:
        """Gate blocks when orange_rate exceeds threshold."""
        result = evaluate_orange_rate_gate(orange_rate=0.12, max_rate=0.10)
        assert result["passed"] is False


class TestLatencyGate:
    """Test latency gate evaluation."""

    def test_latency_gate_passes_below_threshold(self) -> None:
        """Gate passes when latency is below threshold."""
        result = evaluate_latency_gate(latency_ms=900.0, max_latency_ms=1000.0)
        assert result["passed"] is True
        assert result["latency_ms"] == 900.0

    def test_latency_gate_passes_at_threshold(self) -> None:
        """Gate passes when latency equals threshold."""
        result = evaluate_latency_gate(latency_ms=1000.0, max_latency_ms=1000.0)
        assert result["passed"] is True

    def test_latency_gate_blocks_above_threshold(self) -> None:
        """Gate blocks when latency exceeds threshold."""
        result = evaluate_latency_gate(latency_ms=1050.0, max_latency_ms=1000.0)
        assert result["passed"] is False


class TestPromotionGatesIntegration:
    """Integration tests for combined gate evaluation."""

    def test_all_gates_pass(self, feature_ae_gates_config: dict) -> None:
        """All gates pass => no rollback signal."""
        result = evaluate_promotion_gates(
            candidate_recall=1.0,
            candidate_ap=0.94,
            candidate_orange_rate=0.08,
            candidate_latency_ms=900.0,
            prod_ap=0.95,
            gates_config=feature_ae_gates_config,
        )
        assert result["all_passed"] is True
        assert result["rollback_signal"] is False
        assert len(result["gates"]) == 4

    def test_recall_gate_fails_triggers_rollback(
        self, feature_ae_gates_config: dict
    ) -> None:
        """Recall gate failure triggers rollback."""
        result = evaluate_promotion_gates(
            candidate_recall=0.98,  # Below threshold
            candidate_ap=0.94,
            candidate_orange_rate=0.08,
            candidate_latency_ms=900.0,
            prod_ap=0.95,
            gates_config=feature_ae_gates_config,
        )
        assert result["all_passed"] is False
        assert result["rollback_signal"] is True
        assert result["gates"]["recall"]["passed"] is False

    def test_ap_regression_exceeds_triggers_rollback(
        self, feature_ae_gates_config: dict
    ) -> None:
        """AP regression exceeding max triggers rollback."""
        result = evaluate_promotion_gates(
            candidate_recall=1.0,
            candidate_ap=0.92,  # 0.95 - 0.92 = 0.03 regression
            candidate_orange_rate=0.08,
            candidate_latency_ms=900.0,
            prod_ap=0.95,
            gates_config=feature_ae_gates_config,  # candidate AP regresses past 0.02
        )
        assert result["all_passed"] is False
        assert result["rollback_signal"] is True
        assert result["gates"]["ap_regression"]["passed"] is False

    def test_latency_exceeds_triggers_rollback(
        self, feature_ae_gates_config: dict
    ) -> None:
        """Latency exceeding max triggers rollback."""
        result = evaluate_promotion_gates(
            candidate_recall=1.0,
            candidate_ap=0.94,
            candidate_orange_rate=0.08,
            candidate_latency_ms=1100.0,  # Exceeds threshold
            prod_ap=0.95,
            gates_config=feature_ae_gates_config,
        )
        assert result["all_passed"] is False
        assert result["rollback_signal"] is True
        assert result["gates"]["latency"]["passed"] is False
