"""Tests for model promotion workflow (IQA2_KEN09).

Promotion = MLflow state transition (candidate → test/prod) controlled by gates.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from iqa.promotion.promotion import (
    evaluate_gates_for_promotion,
    promote_model_with_gates,
    resolve_model_artifacts,
    transition_model_stage,
)


class TestGateEvaluationForPromotion:
    """Test gate evaluation before promotion decision."""

    def test_promotion_blocked_when_recall_gate_fails(self) -> None:
        """Promotion blocked if recall gate fails (no false negatives allowed)."""
        gates_config = {
            "feature_ae": {
                "recall_defect_min": 1.0,
                "image_ap_max_regression": 0.02,
                "latency_p95_ms_max": 1000,
            }
        }
        candidate_metrics = {
            "recall": 0.99,  # Below required 1.0
            "ap": 0.85,
            "orange_rate": 0.05,
            "latency_ms": 800,
        }

        decision = evaluate_gates_for_promotion(
            registered_model_name="feature_ae__production_replay_natural",
            candidate_metrics=candidate_metrics,
            gates_config=gates_config,
        )

        assert decision["passed"] is False
        assert decision["blocked"] is True
        assert "recall" in decision["blocked_reasons"]

    def test_gates_pass_when_all_metrics_acceptable(self) -> None:
        """Gates pass when all candidate metrics meet thresholds."""
        gates_config = {
            "feature_ae": {
                "recall_defect_min": 1.0,
                "image_ap_max_regression": 0.02,
                "latency_p95_ms_max": 1000,
            }
        }
        candidate_metrics = {
            "recall": 1.0,  # Perfect recall
            "ap": 0.85,
            "orange_rate": 0.05,
            "latency_ms": 800,
        }

        decision = evaluate_gates_for_promotion(
            registered_model_name="feature_ae__production_replay_natural",
            candidate_metrics=candidate_metrics,
            gates_config=gates_config,
        )

        assert decision["passed"] is True
        assert decision["blocked"] is False
        assert decision["blocked_reasons"] == []

    def test_promotion_blocked_when_ap_regresses_too_much(self) -> None:
        """Promotion blocked if AP regresses more than tolerance."""
        gates_config = {
            "feature_ae": {
                "recall_defect_min": 1.0,
                "image_ap_max_regression": 0.02,
                "latency_p95_ms_max": 1000,
            }
        }
        candidate_metrics = {
            "recall": 1.0,
            "ap": 0.90,
            "orange_rate": 0.05,
            "latency_ms": 800,
        }
        prod_metrics = {"ap": 0.95}  # Regression of 0.05, exceeds max 0.02

        decision = evaluate_gates_for_promotion(
            registered_model_name="feature_ae__production_replay_natural",
            candidate_metrics=candidate_metrics,
            gates_config=gates_config,
            prod_metrics=prod_metrics,
        )

        assert decision["passed"] is False
        assert decision["blocked"] is True
        assert "ap_regression" in decision["blocked_reasons"]

    def test_promotion_blocked_when_orange_rate_exceeds_threshold(self) -> None:
        """Promotion blocked if orange rate exceeds maximum."""
        gates_config = {
            "feature_ae": {
                "recall_defect_min": 1.0,
                "orange_rate_max": 0.10,
                "image_ap_max_regression": 0.02,
                "latency_p95_ms_max": 1000,
            }
        }
        candidate_metrics = {
            "recall": 1.0,
            "ap": 0.85,
            "orange_rate": 0.15,  # Exceeds max 0.10
            "latency_ms": 800,
        }

        decision = evaluate_gates_for_promotion(
            registered_model_name="feature_ae__production_replay_natural",
            candidate_metrics=candidate_metrics,
            gates_config=gates_config,
        )

        assert decision["passed"] is False
        assert "orange_rate" in decision["blocked_reasons"]

    def test_promotion_blocked_when_latency_exceeds_threshold(self) -> None:
        """Promotion blocked if latency exceeds maximum."""
        gates_config = {
            "feature_ae": {
                "recall_defect_min": 1.0,
                "image_ap_max_regression": 0.02,
                "latency_p95_ms_max": 1000,
            }
        }
        candidate_metrics = {
            "recall": 1.0,
            "ap": 0.85,
            "orange_rate": 0.05,
            "latency_ms": 1500,  # Exceeds max 1000
        }

        decision = evaluate_gates_for_promotion(
            registered_model_name="feature_ae__production_replay_natural",
            candidate_metrics=candidate_metrics,
            gates_config=gates_config,
        )

        assert decision["passed"] is False
        assert "latency" in decision["blocked_reasons"]

    def test_gates_work_for_different_scenarios(self) -> None:
        """Gate evaluation works for different registered model scenarios."""
        gates_config = {
            "feature_ae": {
                "recall_defect_min": 1.0,
                "image_ap_max_regression": 0.02,
                "latency_p95_ms_max": 1000,
            }
        }
        candidate_metrics = {
            "recall": 1.0,
            "ap": 0.85,
            "orange_rate": 0.05,
            "latency_ms": 800,
        }

        for model_name in [
            "feature_ae__production_replay_natural",
            "feature_ae__drift_domain_extension",
            "roi__surface_defects",
        ]:
            decision = evaluate_gates_for_promotion(
                registered_model_name=model_name,
                candidate_metrics=candidate_metrics,
                gates_config=gates_config,
            )
            assert decision["passed"] is True


class TestMLflowStateTransition:
    """Test MLflow model version stage transitions."""

    @patch("mlflow.tracking.MlflowClient")
    @patch("mlflow.set_tracking_uri")
    def test_transition_model_stage_success(
        self, mock_set_uri: MagicMock, mock_client_class: MagicMock
    ) -> None:
        """Successfully transition model version to new stage."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_model_version = MagicMock()
        mock_model_version.current_stage = "candidate"
        mock_client.transition_model_version_stage.return_value = mock_model_version

        result = transition_model_stage(
            registered_model_name="feature_ae__production_replay_natural",
            version="1",
            target_stage="test",
        )

        assert result["success"] is True
        assert result["new_stage"] == "test"
        assert result["previous_stage"] == "candidate"
        assert result["version"] == "1"
        mock_client.transition_model_version_stage.assert_called_once_with(
            name="feature_ae__production_replay_natural",
            version="1",
            stage="test",
        )

    @patch("mlflow.tracking.MlflowClient")
    @patch("mlflow.set_tracking_uri")
    def test_transition_model_stage_failure(
        self, mock_set_uri: MagicMock, mock_client_class: MagicMock
    ) -> None:
        """Handle transition failure gracefully."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.transition_model_version_stage.side_effect = Exception("MLflow error")

        result = transition_model_stage(
            registered_model_name="feature_ae__production_replay_natural",
            version="1",
            target_stage="test",
        )

        assert result["success"] is False
        assert "error" in result


class TestArtifactResolution:
    """Test resolving model artifacts from MinIO."""

    @patch("mlflow.tracking.MlflowClient")
    @patch("mlflow.set_tracking_uri")
    def test_resolve_artifacts_returns_s3_uri(
        self, mock_set_uri: MagicMock, mock_client_class: MagicMock
    ) -> None:
        """Resolve model artifacts returns S3 URI from MinIO."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_model_version = MagicMock()
        mock_model_version.version = "1"
        mock_model_version.source = "s3://iqa-models/feature_ae__production_replay_natural/1/model"
        mock_client.get_latest_versions.return_value = [mock_model_version]

        result = resolve_model_artifacts(
            registered_model_name="feature_ae__production_replay_natural",
            stage="prod",
        )

        assert result["artifact_uri"] == "s3://iqa-models/feature_ae__production_replay_natural/1/model"
        assert result["stage"] == "prod"
        assert result["registered_model_name"] == "feature_ae__production_replay_natural"
        assert result["version"] == "1"

    @patch("mlflow.tracking.MlflowClient")
    @patch("mlflow.set_tracking_uri")
    def test_resolve_artifacts_raises_on_missing_version(
        self, mock_set_uri: MagicMock, mock_client_class: MagicMock
    ) -> None:
        """Raise error if no model version found in stage."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.get_latest_versions.return_value = []

        with pytest.raises(ValueError, match="No model version found"):
            resolve_model_artifacts(
                registered_model_name="feature_ae__production_replay_natural",
                stage="prod",
            )

    @patch("mlflow.tracking.MlflowClient")
    @patch("mlflow.set_tracking_uri")
    def test_resolve_artifacts_for_test_stage(
        self, mock_set_uri: MagicMock, mock_client_class: MagicMock
    ) -> None:
        """Resolve artifacts for test stage."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_model_version = MagicMock()
        mock_model_version.version = "2"
        mock_model_version.source = "s3://iqa-models/feature_ae__production_replay_natural/2/model"
        mock_client.get_latest_versions.return_value = [mock_model_version]

        result = resolve_model_artifacts(
            registered_model_name="feature_ae__production_replay_natural",
            stage="test",
        )

        assert result["stage"] == "test"
        assert result["version"] == "2"
        mock_client.get_latest_versions.assert_called_once_with(
            "feature_ae__production_replay_natural",
            stages=["test"],
        )

    @patch("mlflow.tracking.MlflowClient")
    @patch("mlflow.set_tracking_uri")
    def test_resolve_artifacts_for_candidate_stage(
        self, mock_set_uri: MagicMock, mock_client_class: MagicMock
    ) -> None:
        """Resolve artifacts for candidate stage."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_model_version = MagicMock()
        mock_model_version.version = "3"
        mock_model_version.source = "s3://mlflow-artifacts/feature_ae__production_replay_natural/3/model"
        mock_client.get_latest_versions.return_value = [mock_model_version]

        result = resolve_model_artifacts(
            registered_model_name="feature_ae__production_replay_natural",
            stage="candidate",
        )

        assert result["stage"] == "candidate"
        assert result["version"] == "3"


class TestProductionPromotionWrapper:
    """Test production-ready promotion wrapper that loads config from file."""

    def test_promote_model_succeeds_when_gates_pass(
        self, tmp_path
    ) -> None:
        """promote_model_with_gates succeeds: load config, check gates, transition, resolve artifacts."""

        gates_config_content = """
feature_ae:
  recall_defect_min: 1.0
  image_ap_max_regression: 0.02
  latency_p95_ms_max: 1000
"""
        config_file = tmp_path / "promotion_gates.yaml"
        config_file.write_text(gates_config_content)

        with patch("mlflow.tracking.MlflowClient") as mock_client_class, patch(
            "mlflow.set_tracking_uri"
        ):
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            mock_model_version = MagicMock()
            mock_model_version.current_stage = "candidate"
            mock_model_version.version = "1"
            mock_model_version.source = (
                "s3://iqa-models/feature_ae__production_replay_natural/1/model"
            )

            mock_client.transition_model_version_stage.return_value = mock_model_version
            mock_client.get_latest_versions.return_value = [mock_model_version]

            result = promote_model_with_gates(
                registered_model_name="feature_ae__production_replay_natural",
                version="1",
                target_stage="test",
                candidate_metrics={
                    "recall": 1.0,
                    "ap": 0.85,
                    "orange_rate": 0.05,
                    "latency_ms": 800,
                },
                gates_config_path=str(config_file),
            )

            assert result["success"] is True
            assert result["gates_passed"] is True
            assert result["transition"]["success"] is True
            assert result["artifacts"]["stage"] == "test"

    def test_promote_model_blocked_when_gates_fail(self, tmp_path) -> None:
        """promote_model_with_gates blocks promotion when gates fail (no MLflow transition)."""

        gates_config_content = """
feature_ae:
  recall_defect_min: 1.0
  image_ap_max_regression: 0.02
  latency_p95_ms_max: 1000
"""
        config_file = tmp_path / "promotion_gates.yaml"
        config_file.write_text(gates_config_content)

        with patch("mlflow.tracking.MlflowClient") as mock_client_class, patch(
            "mlflow.set_tracking_uri"
        ):
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            result = promote_model_with_gates(
                registered_model_name="feature_ae__production_replay_natural",
                version="1",
                target_stage="test",
                candidate_metrics={
                    "recall": 0.99,  # Below required 1.0
                    "ap": 0.85,
                    "orange_rate": 0.05,
                    "latency_ms": 800,
                },
                gates_config_path=str(config_file),
            )

            assert result["success"] is False
            assert result["gates_passed"] is False
            assert "recall" in result["blocked_reasons"]
            mock_client.transition_model_version_stage.assert_not_called()
