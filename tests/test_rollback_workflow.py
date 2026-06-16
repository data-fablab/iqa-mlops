"""Tests for model rollback workflow (IQA2_KEN10).

Rollback = restore previous_prod to prod stage, archive faulty version.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from iqa.promotion.rollback import (
    get_previous_prod,
    rollback_model,
    save_previous_prod_before_promotion,
)


class TestSavePreviousProd:
    """Test saving current prod version before promotion."""

    @patch("mlflow.tracking.MlflowClient")
    @patch("mlflow.set_tracking_uri")
    def test_save_previous_prod_records_current_prod_version(
        self, mock_set_uri: MagicMock, mock_client_class: MagicMock
    ) -> None:
        """save_previous_prod_before_promotion saves current prod version to MLflow."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_prod_version = MagicMock()
        mock_prod_version.version = "5"
        mock_client.get_latest_versions.return_value = [mock_prod_version]

        result = save_previous_prod_before_promotion(
            registered_model_name="feature_ae__production_replay_natural"
        )

        assert result["success"] is True
        assert result["previous_prod_version"] == "5"
        assert result["registered_model_name"] == "feature_ae__production_replay_natural"
        mock_client.get_latest_versions.assert_called_once_with(
            "feature_ae__production_replay_natural",
            stages=["prod"],
        )
        # The "previous_prod" alias MUST be persisted so rollback can find it.
        mock_client.set_registered_model_alias.assert_called_once_with(
            name="feature_ae__production_replay_natural",
            alias="previous_prod",
            version="5",
        )

    @patch("mlflow.tracking.MlflowClient")
    @patch("mlflow.set_tracking_uri")
    def test_save_previous_prod_raises_when_no_prod_exists(
        self, mock_set_uri: MagicMock, mock_client_class: MagicMock
    ) -> None:
        """save_previous_prod raises error if no prod version exists."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.get_latest_versions.return_value = []

        result = save_previous_prod_before_promotion(
            registered_model_name="feature_ae__production_replay_natural"
        )

        assert result["success"] is False
        assert "error" in result


class TestGetPreviousProd:
    """Test retrieving saved previous_prod version."""

    @patch("mlflow.tracking.MlflowClient")
    @patch("mlflow.set_tracking_uri")
    def test_get_previous_prod_returns_version(
        self, mock_set_uri: MagicMock, mock_client_class: MagicMock
    ) -> None:
        """get_previous_prod retrieves saved previous_prod version from alias."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_model_version = MagicMock()
        mock_model_version.version = "4"
        mock_client.get_model_version_by_alias.return_value = mock_model_version

        result = get_previous_prod(
            registered_model_name="feature_ae__production_replay_natural"
        )

        assert result["version"] == "4"
        assert result["registered_model_name"] == "feature_ae__production_replay_natural"
        mock_client.get_model_version_by_alias.assert_called_once_with(
            "feature_ae__production_replay_natural",
            "previous_prod",
        )

    @patch("mlflow.tracking.MlflowClient")
    @patch("mlflow.set_tracking_uri")
    def test_get_previous_prod_raises_when_not_found(
        self, mock_set_uri: MagicMock, mock_client_class: MagicMock
    ) -> None:
        """get_previous_prod raises error if previous_prod alias not found."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.get_model_version_by_alias.side_effect = Exception("Not found")

        with pytest.raises(ValueError, match="No previous_prod version found"):
            get_previous_prod(
                registered_model_name="feature_ae__production_replay_natural"
            )


class TestRollbackModel:
    """Test rolling back to previous production version."""

    @patch("iqa.promotion.rollback.get_previous_prod")
    @patch("mlflow.tracking.MlflowClient")
    @patch("mlflow.set_tracking_uri")
    def test_rollback_restores_previous_prod_and_archives_faulty(
        self,
        mock_set_uri: MagicMock,
        mock_client_class: MagicMock,
        mock_get_prev: MagicMock,
    ) -> None:
        """rollback_model restores previous_prod to prod and archives faulty version."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_get_prev.return_value = {"version": "4"}

        result = rollback_model(
            registered_model_name="feature_ae__production_replay_natural",
            faulty_version="6",
        )

        assert result["success"] is True
        assert result["previous_prod_version"] == "4"
        assert result["faulty_version_archived"] == "6"

        # Verify transitions were called
        calls = mock_client.transition_model_version_stage.call_args_list
        assert len(calls) == 2
        # First call: restore previous_prod to prod
        assert calls[0][1]["version"] == "4"
        assert calls[0][1]["stage"] == "prod"
        # Second call: archive faulty
        assert calls[1][1]["version"] == "6"
        assert calls[1][1]["stage"] == "archived"

    @patch("iqa.promotion.rollback.get_previous_prod")
    @patch("mlflow.tracking.MlflowClient")
    @patch("mlflow.set_tracking_uri")
    def test_rollback_raises_when_no_previous_prod(
        self,
        mock_set_uri: MagicMock,
        mock_client_class: MagicMock,
        mock_get_prev: MagicMock,
    ) -> None:
        """rollback_model fails if no previous_prod is available."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_get_prev.side_effect = ValueError("No previous_prod version found")

        result = rollback_model(
            registered_model_name="feature_ae__production_replay_natural",
            faulty_version="6",
        )

        assert result["success"] is False
        assert "error" in result

    @patch("iqa.promotion.rollback.get_previous_prod")
    @patch("mlflow.tracking.MlflowClient")
    @patch("mlflow.set_tracking_uri")
    def test_rollback_works_for_different_scenarios(
        self,
        mock_set_uri: MagicMock,
        mock_client_class: MagicMock,
        mock_get_prev: MagicMock,
    ) -> None:
        """rollback_model works for different registered model scenarios."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_get_prev.return_value = {"version": "2"}

        for model_name in [
            "feature_ae__production_replay_natural",
            "feature_ae__drift_domain_extension",
            "roi__surface_defects",
        ]:
            result = rollback_model(
                registered_model_name=model_name,
                faulty_version="3",
            )
            assert result["success"] is True
            assert result["previous_prod_version"] == "2"


class TestRollbackIntegration:
    """Integration tests for promotion + rollback workflow."""

    @patch("iqa.promotion.rollback.get_previous_prod")
    @patch("mlflow.tracking.MlflowClient")
    @patch("mlflow.set_tracking_uri")
    def test_full_workflow_save_then_rollback(
        self,
        mock_set_uri: MagicMock,
        mock_client_class: MagicMock,
        mock_get_prev: MagicMock,
    ) -> None:
        """Full workflow: save prod before promotion, rollback if needed."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Step 1: Save current prod (version 5)
        mock_prod_version = MagicMock()
        mock_prod_version.version = "5"
        mock_client.get_latest_versions.return_value = [mock_prod_version]

        save_result = save_previous_prod_before_promotion(
            registered_model_name="feature_ae__production_replay_natural"
        )
        assert save_result["success"] is True
        assert save_result["previous_prod_version"] == "5"

        # Step 2: Rollback from faulty version 7 back to version 5
        mock_get_prev.return_value = {"version": "5"}

        rollback_result = rollback_model(
            registered_model_name="feature_ae__production_replay_natural",
            faulty_version="7",
        )
        assert rollback_result["success"] is True
        assert rollback_result["previous_prod_version"] == "5"
        assert rollback_result["faulty_version_archived"] == "7"

    @patch("iqa.promotion.rollback.get_previous_prod")
    @patch("mlflow.tracking.MlflowClient")
    @patch("mlflow.set_tracking_uri")
    def test_multiple_rollbacks_same_model(
        self,
        mock_set_uri: MagicMock,
        mock_client_class: MagicMock,
        mock_get_prev: MagicMock,
    ) -> None:
        """Can perform multiple rollbacks on the same model."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Rollback from version 7
        mock_get_prev.return_value = {"version": "5"}
        result1 = rollback_model(
            registered_model_name="feature_ae__production_replay_natural",
            faulty_version="7",
        )
        assert result1["success"] is True

        # Rollback from version 8 (if another promotion fails)
        mock_get_prev.return_value = {"version": "5"}
        result2 = rollback_model(
            registered_model_name="feature_ae__production_replay_natural",
            faulty_version="8",
        )
        assert result2["success"] is True
        assert result2["faulty_version_archived"] == "8"
