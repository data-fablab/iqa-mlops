"""Tests for promotion/rollback cycle: gate blocking and model recovery (IQA2_KEN16)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from iqa.promotion.promotion import evaluate_gates_for_promotion, promote_model_with_gates
from iqa.promotion.rollback import (
    get_previous_prod,
    rollback_model,
    save_previous_prod_before_promotion,
)


class TestFalseNegativeBlocksPromotion:
    """Verify false negative (recall < 1.0) blocks promotion."""

    def test_false_negative_blocks_promotion(self) -> None:
        """False negative blocks promotion: recall must be 1.0."""
        gates_config = {
            "feature_ae": {
                "recall_defect_min": 1.0,
                "image_ap_max_regression": 0.02,
                "orange_rate_max": 0.10,
                "latency_p95_ms_max": 1000,
            }
        }
        candidate_metrics = {
            "recall": 0.98,  # Missing 2% of defects (false negatives)
            "ap": 0.95,
            "orange_rate": 0.05,
            "latency_ms": 800,
        }

        decision = evaluate_gates_for_promotion(
            registered_model_name="feature_ae__production_replay_natural",
            candidate_metrics=candidate_metrics,
            gates_config=gates_config,
        )

        assert decision["blocked"] is True
        assert "recall" in decision["blocked_reasons"]
        assert decision["passed"] is False

    def test_recall_gate_passes_only_at_perfect_recall(self) -> None:
        """Recall gate passes only when recall == 1.0."""
        gates_config = {
            "feature_ae": {
                "recall_defect_min": 1.0,
                "image_ap_max_regression": 0.02,
                "orange_rate_max": 0.10,
                "latency_p95_ms_max": 1000,
            }
        }

        # Test at perfect recall
        candidate_metrics = {
            "recall": 1.0,  # Perfect recall
            "ap": 0.95,
            "orange_rate": 0.05,
            "latency_ms": 800,
        }
        decision = evaluate_gates_for_promotion(
            registered_model_name="feature_ae__production_replay_natural",
            candidate_metrics=candidate_metrics,
            gates_config=gates_config,
        )
        assert decision["passed"] is True

    def test_even_small_false_negative_rate_blocks(self) -> None:
        """Even 0.1% false negative rate blocks promotion."""
        gates_config = {
            "feature_ae": {
                "recall_defect_min": 1.0,
                "image_ap_max_regression": 0.02,
                "orange_rate_max": 0.10,
                "latency_p95_ms_max": 1000,
            }
        }
        candidate_metrics = {
            "recall": 0.999,  # 0.1% false negative rate
            "ap": 0.95,
            "orange_rate": 0.05,
            "latency_ms": 800,
        }

        decision = evaluate_gates_for_promotion(
            registered_model_name="feature_ae__production_replay_natural",
            candidate_metrics=candidate_metrics,
            gates_config=gates_config,
        )

        assert decision["blocked"] is True
        assert "recall" in decision["blocked_reasons"]


class TestInsufficientAPBlocksPromotion:
    """Verify insufficient AP (regression > 0.02) blocks promotion."""

    def test_ap_regression_exceeding_tolerance_blocks_promotion(self) -> None:
        """AP regression > 0.02 blocks promotion."""
        gates_config = {
            "feature_ae": {
                "recall_defect_min": 1.0,
                "image_ap_max_regression": 0.02,
                "orange_rate_max": 0.10,
                "latency_p95_ms_max": 1000,
            }
        }
        candidate_metrics = {
            "recall": 1.0,
            "ap": 0.92,  # Regression of 0.03 from prod_ap=0.95
            "orange_rate": 0.05,
            "latency_ms": 800,
        }
        prod_metrics = {"ap": 0.95}

        decision = evaluate_gates_for_promotion(
            registered_model_name="feature_ae__production_replay_natural",
            candidate_metrics=candidate_metrics,
            gates_config=gates_config,
            prod_metrics=prod_metrics,
        )

        assert decision["blocked"] is True
        assert "ap_regression" in decision["blocked_reasons"]
        assert decision["passed"] is False

    def test_ap_at_exact_regression_threshold_passes(self) -> None:
        """AP at exact regression threshold (0.02) passes."""
        gates_config = {
            "feature_ae": {
                "recall_defect_min": 1.0,
                "image_ap_max_regression": 0.02,
                "orange_rate_max": 0.10,
                "latency_p95_ms_max": 1000,
            }
        }
        candidate_metrics = {
            "recall": 1.0,
            "ap": 0.93,  # Exactly 0.02 regression from 0.95
            "orange_rate": 0.05,
            "latency_ms": 800,
        }
        prod_metrics = {"ap": 0.95}

        decision = evaluate_gates_for_promotion(
            registered_model_name="feature_ae__production_replay_natural",
            candidate_metrics=candidate_metrics,
            gates_config=gates_config,
            prod_metrics=prod_metrics,
        )

        assert decision["passed"] is True
        assert "ap_regression" not in decision["blocked_reasons"]

    def test_ap_improvement_passes_gate(self) -> None:
        """AP improvement (negative regression) passes gate."""
        gates_config = {
            "feature_ae": {
                "recall_defect_min": 1.0,
                "image_ap_max_regression": 0.02,
                "orange_rate_max": 0.10,
                "latency_p95_ms_max": 1000,
            }
        }
        candidate_metrics = {
            "recall": 1.0,
            "ap": 0.97,  # Improvement from 0.95
            "orange_rate": 0.05,
            "latency_ms": 800,
        }
        prod_metrics = {"ap": 0.95}

        decision = evaluate_gates_for_promotion(
            registered_model_name="feature_ae__production_replay_natural",
            candidate_metrics=candidate_metrics,
            gates_config=gates_config,
            prod_metrics=prod_metrics,
        )

        assert decision["passed"] is True


class TestPromotionSucceedsWhenGatesPass:
    """Verify promotion succeeds when all gates pass."""

    @patch("iqa.promotion.promotion.transition_model_stage")
    @patch("iqa.promotion.promotion.resolve_model_artifacts")
    @patch("builtins.open")
    def test_promotion_succeeds_when_all_gates_pass(
        self,
        mock_open: MagicMock,
        mock_resolve: MagicMock,
        mock_transition: MagicMock,
    ) -> None:
        """Promotion succeeds: gates pass → transition → resolve artifacts."""
        mock_config_file = MagicMock()
        mock_config_file.__enter__.return_value.read.return_value = """
feature_ae:
  recall_defect_min: 1.0
  image_ap_max_regression: 0.02
  orange_rate_max: 0.10
  latency_p95_ms_max: 1000
"""
        mock_open.return_value = mock_config_file

        mock_transition.return_value = {
            "success": True,
            "new_stage": "prod",
            "previous_stage": "candidate",
        }
        mock_resolve.return_value = {
            "artifact_uri": "mlruns/0/abc123/artifacts",
            "version": "5",
        }

        result = promote_model_with_gates(
            registered_model_name="feature_ae__production_replay_natural",
            version="5",
            target_stage="prod",
            candidate_metrics={
                "recall": 1.0,
                "ap": 0.95,
                "orange_rate": 0.05,
                "latency_ms": 800,
            },
            prod_metrics={"ap": 0.92},
        )

        assert result["success"] is True
        assert result["gates_passed"] is True
        assert result["transition"]["success"] is True

    @patch("builtins.open")
    def test_promotion_blocked_when_gates_fail(self, mock_open: MagicMock) -> None:
        """Promotion fails: gates fail → no transition."""
        mock_config_file = MagicMock()
        mock_config_file.__enter__.return_value.read.return_value = """
feature_ae:
  recall_defect_min: 1.0
  image_ap_max_regression: 0.02
  orange_rate_max: 0.10
  latency_p95_ms_max: 1000
"""
        mock_open.return_value = mock_config_file

        result = promote_model_with_gates(
            registered_model_name="feature_ae__production_replay_natural",
            version="5",
            target_stage="prod",
            candidate_metrics={
                "recall": 0.99,  # Fails recall gate
                "ap": 0.95,
                "orange_rate": 0.05,
                "latency_ms": 800,
            },
            prod_metrics={"ap": 0.92},
        )

        assert result["success"] is False
        assert result["gates_passed"] is False
        assert "recall" in result["blocked_reasons"]


class TestRollbackRestoresPreviousProd:
    """Verify rollback restores previous_prod to production."""

    @patch("mlflow.tracking.MlflowClient")
    @patch("mlflow.set_tracking_uri")
    def test_rollback_restores_previous_prod(
        self, mock_set_uri: MagicMock, mock_client_class: MagicMock
    ) -> None:
        """Rollback restores previous_prod to prod and archives faulty."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock get_model_version_by_alias for previous_prod lookup
        mock_prev_version = MagicMock()
        mock_prev_version.version = "4"
        mock_client.get_model_version_by_alias.return_value = mock_prev_version

        result = rollback_model(
            registered_model_name="feature_ae__production_replay_natural",
            faulty_version="5",
        )

        assert result["success"] is True
        assert result["previous_prod_version"] == "4"
        assert result["faulty_version_archived"] == "5"

        # Verify transitions were called correctly
        transition_calls = mock_client.transition_model_version_stage.call_args_list
        assert len(transition_calls) == 2

        # First call: restore previous_prod to prod
        assert transition_calls[0][1]["stage"] == "prod"
        assert transition_calls[0][1]["version"] == "4"

        # Second call: archive faulty
        assert transition_calls[1][1]["stage"] == "archived"
        assert transition_calls[1][1]["version"] == "5"

    @patch("mlflow.tracking.MlflowClient")
    @patch("mlflow.set_tracking_uri")
    def test_rollback_saves_and_restores_workflow(
        self, mock_set_uri: MagicMock, mock_client_class: MagicMock
    ) -> None:
        """Full workflow: save previous_prod before promotion, then rollback."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Phase 1: Save current prod before promotion
        mock_prod_version = MagicMock()
        mock_prod_version.version = "4"
        mock_client.get_latest_versions.return_value = [mock_prod_version]

        save_result = save_previous_prod_before_promotion(
            registered_model_name="feature_ae__production_replay_natural"
        )
        assert save_result["success"] is True
        assert save_result["previous_prod_version"] == "4"

        # Phase 2: New model promoted (v5 → prod) — not in this test
        # In real scenario, v5 would be promoted and something goes wrong

        # Phase 3: Rollback triggered (restore v4, archive v5)
        mock_prev_version = MagicMock()
        mock_prev_version.version = "4"
        mock_client.get_model_version_by_alias.return_value = mock_prev_version

        rollback_result = rollback_model(
            registered_model_name="feature_ae__production_replay_natural",
            faulty_version="5",
        )
        assert rollback_result["success"] is True
        assert rollback_result["previous_prod_version"] == "4"

    def test_multiple_rollback_attempts(self) -> None:
        """Multiple rollback attempts work correctly for different faulty versions."""
        with patch("mlflow.tracking.MlflowClient") as mock_client_class, \
             patch("mlflow.set_tracking_uri"):

            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # First rollback: v5 was faulty, restore v4
            mock_prev_v1 = MagicMock()
            mock_prev_v1.version = "4"
            mock_client.get_model_version_by_alias.return_value = mock_prev_v1

            result1 = rollback_model(
                registered_model_name="feature_ae__production_replay_natural",
                faulty_version="5",
            )
            assert result1["previous_prod_version"] == "4"
            assert result1["faulty_version_archived"] == "5"

            # Second rollback: v6 was faulty, restore v4 again
            # (simulating multiple attempts with same previous_prod)
            result2 = rollback_model(
                registered_model_name="feature_ae__production_replay_natural",
                faulty_version="6",
            )
            assert result2["previous_prod_version"] == "4"
            assert result2["faulty_version_archived"] == "6"


class TestPromotionRollbackCycleIntegration:
    """End-to-end promotion and rollback cycle."""

    def test_full_cycle_gate_block_prevents_need_for_rollback(self) -> None:
        """Good gates prevent bad models from being promoted (no rollback needed)."""
        gates_config = {
            "feature_ae": {
                "recall_defect_min": 1.0,
                "image_ap_max_regression": 0.02,
                "orange_rate_max": 0.10,
                "latency_p95_ms_max": 1000,
            }
        }

        # Candidate with recall = 0.99 should be blocked
        decision = evaluate_gates_for_promotion(
            registered_model_name="feature_ae__production_replay_natural",
            candidate_metrics={
                "recall": 0.99,  # Fails
                "ap": 0.95,
                "orange_rate": 0.05,
                "latency_ms": 800,
            },
            gates_config=gates_config,
            prod_metrics={"ap": 0.95},
        )

        assert decision["blocked"] is True
        # Promotion never happens, so rollback never needed

    @patch("mlflow.tracking.MlflowClient")
    @patch("mlflow.set_tracking_uri")
    def test_gates_block_multiple_failure_scenarios(
        self, mock_set_uri: MagicMock, mock_client_class: MagicMock
    ) -> None:
        """Gates block multiple failure scenarios (FN, AP, latency)."""
        gates_config = {
            "feature_ae": {
                "recall_defect_min": 1.0,
                "image_ap_max_regression": 0.02,
                "orange_rate_max": 0.10,
                "latency_p95_ms_max": 1000,
            }
        }

        scenarios = [
            (
                {"recall": 0.98, "ap": 0.95, "orange_rate": 0.05, "latency_ms": 800},
                ["recall"],
                "FN blocks",
            ),
            (
                {"recall": 1.0, "ap": 0.90, "orange_rate": 0.05, "latency_ms": 800},
                ["ap_regression"],
                "AP regression blocks",
            ),
            (
                {"recall": 1.0, "ap": 0.95, "orange_rate": 0.15, "latency_ms": 800},
                ["orange_rate"],
                "Orange rate blocks",
            ),
            (
                {"recall": 1.0, "ap": 0.95, "orange_rate": 0.05, "latency_ms": 1100},
                ["latency"],
                "Latency blocks",
            ),
        ]

        for metrics, expected_blocked, description in scenarios:
            decision = evaluate_gates_for_promotion(
                registered_model_name="feature_ae__production_replay_natural",
                candidate_metrics=metrics,
                gates_config=gates_config,
                prod_metrics={"ap": 0.95},
            )

            assert decision["blocked"] is True, f"Failed: {description}"
            for gate in expected_blocked:
                assert gate in decision["blocked_reasons"], f"Missing gate in {description}"


__all__ = [
    "TestFalseNegativeBlocksPromotion",
    "TestInsufficientAPBlocksPromotion",
    "TestPromotionSucceedsWhenGatesPass",
    "TestRollbackRestoresPreviousProd",
    "TestPromotionRollbackCycleIntegration",
]
