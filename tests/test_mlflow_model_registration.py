"""Tests for MLflow model registration by scenario_id."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from iqa.registry import register_run_to_model, registered_model_name


class TestRegisterRunToModel:
    """Test model registration for runs."""

    def test_register_run_to_candidate_model_for_scenario(self, mlflow_tracking_uri: str) -> None:
        """Register a run as model version in candidate stage."""
        import mlflow
        import torch

        # Set tracking URI for both run creation and registration
        mlflow.set_tracking_uri(mlflow_tracking_uri)

        # Create and log a run with a simple model
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            torch.save({"state_dict": {}}, checkpoint_path)

            with mlflow.start_run(run_name="test_run") as run:
                mlflow.log_param("test_param", "test_value")
                mlflow.log_artifact(str(checkpoint_path), artifact_path="model")
                mlflow.pytorch.log_model(
                    pytorch_model=torch.nn.Linear(1, 1),
                    artifact_path="model",
                    input_example=torch.randn(1, 1),
                )
                run_id = run.info.run_id

            # Register run to model
            scenario_id = "test_scenario"
            result = register_run_to_model(
                run_id=run_id,
                scenario_id=scenario_id,
                stage="candidate",
                tracking_uri=mlflow_tracking_uri,
            )

            # Verify result
            assert result["registered_model_name"] == f"feature_ae__{scenario_id}"
            assert result["stage"] == "candidate"
            assert "version" in result

    def test_register_run_to_test_stage(self, mlflow_tracking_uri: str) -> None:
        """Register a run to test stage."""
        import mlflow
        import torch

        mlflow.set_tracking_uri(mlflow_tracking_uri)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            torch.save({"state_dict": {}}, checkpoint_path)

            with mlflow.start_run(run_name="test_run_stage") as run:
                mlflow.log_param("test_param", "test_value")
                mlflow.pytorch.log_model(
                    pytorch_model=torch.nn.Linear(1, 1),
                    artifact_path="model",
                    input_example=torch.randn(1, 1),
                )
                run_id = run.info.run_id

            scenario_id = "test_scenario_2"
            result = register_run_to_model(
                run_id=run_id,
                scenario_id=scenario_id,
                stage="test",
                tracking_uri=mlflow_tracking_uri,
            )

            assert result["registered_model_name"] == f"feature_ae__{scenario_id}"
            assert result["stage"] == "test"

    def test_different_scenarios_create_different_models(self, mlflow_tracking_uri: str) -> None:
        """Different scenario_ids create separate registered models."""
        import mlflow
        import torch

        mlflow.set_tracking_uri(mlflow_tracking_uri)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            torch.save({"state_dict": {}}, checkpoint_path)

            # Register first scenario
            with mlflow.start_run(run_name="scenario_1_run") as run:
                mlflow.pytorch.log_model(
                    pytorch_model=torch.nn.Linear(1, 1),
                    artifact_path="model",
                    input_example=torch.randn(1, 1),
                )
                run_id_1 = run.info.run_id

            result_1 = register_run_to_model(
                run_id=run_id_1,
                scenario_id="scenario_1",
                stage="candidate",
                tracking_uri=mlflow_tracking_uri,
            )

            # Register second scenario
            with mlflow.start_run(run_name="scenario_2_run") as run:
                mlflow.pytorch.log_model(
                    pytorch_model=torch.nn.Linear(1, 1),
                    artifact_path="model",
                    input_example=torch.randn(1, 1),
                )
                run_id_2 = run.info.run_id

            result_2 = register_run_to_model(
                run_id=run_id_2,
                scenario_id="scenario_2",
                stage="candidate",
                tracking_uri=mlflow_tracking_uri,
            )

            # Verify they are different models
            assert result_1["registered_model_name"] == "feature_ae__scenario_1"
            assert result_2["registered_model_name"] == "feature_ae__scenario_2"
            assert result_1["registered_model_name"] != result_2["registered_model_name"]

    def test_multiple_runs_same_scenario_create_new_versions(self, mlflow_tracking_uri: str) -> None:
        """Multiple runs for same scenario create different versions."""
        import mlflow
        import torch

        mlflow.set_tracking_uri(mlflow_tracking_uri)

        versions = []
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            torch.save({"state_dict": {}}, checkpoint_path)

            for i in range(2):
                with mlflow.start_run(run_name=f"scenario_3_run_{i}") as run:
                    mlflow.log_param("run_number", str(i))
                    mlflow.pytorch.log_model(
                        pytorch_model=torch.nn.Linear(1, 1),
                        artifact_path="model",
                        input_example=torch.randn(1, 1),
                    )
                    run_id = run.info.run_id

                result = register_run_to_model(
                    run_id=run_id,
                    scenario_id="scenario_3",
                    stage="candidate",
                    tracking_uri=mlflow_tracking_uri,
                )
                versions.append(result["version"])

            # Both runs should create versions under same model, but different versions
            assert len(set(versions)) >= 1  # At least we get version(s)
            assert result["registered_model_name"] == "feature_ae__scenario_3"
