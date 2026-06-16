"""Integration tests for the promotion/rollback cycle (IQA2_KEN16).

This module covers ONLY the end-to-end cycle wiring. The per-gate blocking rules
and promote_model_with_gates live in tests/test_promotion_workflow.py; the rollback
mechanics (alias persistence, transitions) live in tests/test_rollback_workflow.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


from iqa.promotion.promotion import evaluate_gates_for_promotion
from iqa.promotion.rollback import (
    rollback_model,
    save_previous_prod_before_promotion,
)


class TestPromotionRollbackCycle:
    """End-to-end promotion and rollback cycle."""

    def test_gate_block_prevents_promotion_so_no_rollback_needed(
        self, feature_ae_gates_config: dict
    ) -> None:
        """A blocked candidate never reaches prod, so no rollback is needed."""
        decision = evaluate_gates_for_promotion(
            registered_model_name="feature_ae__production_replay_natural",
            candidate_metrics={
                "recall": 0.99,  # fails recall gate
                "ap": 0.95,
                "orange_rate": 0.05,
                "latency_ms": 800,
            },
            gates_config=feature_ae_gates_config,
            prod_metrics={"ap": 0.95},
        )

        assert decision["blocked"] is True

    @patch("mlflow.tracking.MlflowClient")
    @patch("mlflow.set_tracking_uri")
    def test_save_previous_prod_then_rollback_restores_it(
        self, mock_set_uri: MagicMock, mock_client_class: MagicMock
    ) -> None:
        """Full cycle: save current prod, promote, then rollback to the saved version."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Phase 1: save current prod (v4) before promoting v5.
        mock_prod_version = MagicMock()
        mock_prod_version.version = "4"
        mock_client.get_latest_versions.return_value = [mock_prod_version]

        save_result = save_previous_prod_before_promotion(
            registered_model_name="feature_ae__production_replay_natural"
        )
        assert save_result["success"] is True
        assert save_result["previous_prod_version"] == "4"

        # Phase 2: v5 promoted then found faulty -> rollback restores v4, archives v5.
        mock_prev_version = MagicMock()
        mock_prev_version.version = "4"
        mock_client.get_model_version_by_alias.return_value = mock_prev_version

        rollback_result = rollback_model(
            registered_model_name="feature_ae__production_replay_natural",
            faulty_version="5",
        )
        assert rollback_result["success"] is True
        assert rollback_result["previous_prod_version"] == "4"
        assert rollback_result["faulty_version_archived"] == "5"


__all__ = ["TestPromotionRollbackCycle"]
