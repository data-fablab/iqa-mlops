"""Tests for MLflow run logging with full traceability."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import torch

from iqa.training.feature_ae import FeatureAETrainingConfig
from iqa.training.feature_ae_evaluation import EvaluationReport
from iqa.training.mlflow_logging import (
    MLflowRunLogger,
    log_promoted_feature_ae_bundle,
    train_feature_ae_with_mlflow_logging,
)


class TestMLflowRunLoggerBasic:
    """Tracer bullet: MLflowRunLogger creates and manages runs."""

    def test_create_run_and_end(self, mlflow_tracking_uri: str) -> None:
        """Test that logger can create and end a run."""
        logger = MLflowRunLogger(
            run_name="test_run",
            scenario_id="test_scenario",
            tracking_uri=mlflow_tracking_uri,
        )

        # End run and verify run_id is returned
        run_id = logger.end_run()
        assert run_id is not None
        assert isinstance(run_id, str)
        assert len(run_id) > 0

    def test_log_params_from_config(
        self, tmp_path: Path, mlflow_tracking_uri: str
    ) -> None:
        """Test that logger logs training params from config."""
        config = FeatureAETrainingConfig(
            manifest_path=tmp_path / "manifest.json",
            image_root=tmp_path / "images",
            output_checkpoint=tmp_path / "checkpoint.pt",
            batch_size=16,
            epochs=10,
            learning_rate=5e-5,
            scenario_id="test_scenario",
            dataset_version="v1",
            manifest_version="v1_manifest_v001",
        )

        logger = MLflowRunLogger(
            run_name="test_run",
            scenario_id="test_scenario",
            tracking_uri=mlflow_tracking_uri,
        )

        logger.log_config(config)

        # Params should be logged
        run_id = logger.end_run()
        assert run_id is not None

    def test_log_metrics(self, mlflow_tracking_uri: str) -> None:
        """Test that logger logs metrics at each step."""
        logger = MLflowRunLogger(
            run_name="test_run",
            scenario_id="test_scenario",
            tracking_uri=mlflow_tracking_uri,
        )

        metrics = {"train_loss": 0.5, "val_loss": 0.6}
        logger.log_metrics(metrics, step=1)
        logger.log_metrics({"train_loss": 0.3, "val_loss": 0.4}, step=2)

        run_id = logger.end_run()
        assert run_id is not None

    def test_log_evaluation_metrics(self, mlflow_tracking_uri: str) -> None:
        """Test that logger logs evaluation metrics (AP, recall, orange_rate, latency)."""
        logger = MLflowRunLogger(
            run_name="test_run",
            scenario_id="test_scenario",
            tracking_uri=mlflow_tracking_uri,
        )

        eval_report = EvaluationReport(
            model_version="v1",
            average_precision=0.95,
            recall=0.92,
            orange_rate=0.05,
            latency_ms=45.2,
            sample_count=100,
        )
        logger.log_evaluation_metrics(eval_report)

        run_id = logger.end_run()
        assert run_id is not None

    def test_log_artifacts(self, tmp_path: Path, mlflow_tracking_uri: str) -> None:
        """Test that logger logs model checkpoint and evaluation report artifacts."""
        logger = MLflowRunLogger(
            run_name="test_run",
            scenario_id="test_scenario",
            tracking_uri=mlflow_tracking_uri,
        )

        # Create dummy checkpoint
        checkpoint_path = tmp_path / "checkpoint.pt"
        torch.save({"state_dict": {}}, checkpoint_path)

        # Create dummy eval report
        report_path = tmp_path / "eval_report.json"
        report_path.write_text(json.dumps({"ap": 0.95}))

        logger.log_artifacts(
            checkpoint_path=checkpoint_path, eval_report_path=report_path
        )

        run_id = logger.end_run()
        assert run_id is not None

    def test_log_feature_ae_model_writes_mlflow_model(
        self,
        tmp_path: Path,
        mlflow_tracking_uri: str,
    ) -> None:
        """Feature-AE logging creates a real MLflow 3 LoggedModel."""
        import mlflow

        config = FeatureAETrainingConfig(
            manifest_path=tmp_path / "manifest.csv",
            image_root=tmp_path / "images",
            output_checkpoint=tmp_path / "checkpoint.pt",
            scenario_id="test_scenario",
            dataset_version="dataset_v1",
            candidate_version="candidate_v1",
        )
        checkpoint_path = tmp_path / "checkpoint.pt"
        torch.save({"state_dict": {}}, checkpoint_path)

        logger = MLflowRunLogger(
            run_name="model_logging_test",
            scenario_id="test_scenario",
            tracking_uri=mlflow_tracking_uri,
        )

        assert (
            logger.log_feature_ae_model(
                config,
                checkpoint_path,
            )
            is True
        )
        run_id = logger.end_run()

        client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_tracking_uri)
        models = client.search_logged_models(
            experiment_ids=["0"],
            max_results=100,
        )
        matches = [
            model
            for model in models
            if str(model.source_run_id) == run_id and model.name == "model"
        ]

        assert len(matches) == 1

        model_id = matches[0].model_id
        root = client.list_logged_model_artifacts(model_id)
        root_paths = {item.path for item in root}

        assert "MLmodel" in root_paths
        assert "artifacts" in root_paths

        artifact_paths = {
            item.path
            for item in client.list_logged_model_artifacts(
                model_id,
                "artifacts",
            )
        }
        assert "artifacts/checkpoint.pt" in artifact_paths
        assert "artifacts/score_contract.json" in artifact_paths

    def test_log_promoted_bundle_is_complete(
        self,
        tmp_path: Path,
        mlflow_tracking_uri: str,
    ) -> None:
        """The promoted serving bundle contains every runtime artifact."""
        import hashlib

        import mlflow

        checkpoint_path = tmp_path / "checkpoint.pt"
        torch.save({"state_dict": {}}, checkpoint_path)

        digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

        manifest = json.loads(
            Path(
                "models/manifests/"
                "rd_feature_ae_gated_v001_bootstrap/"
                "model_manifest.json"
            ).read_text(encoding="utf-8")
        )
        manifest["model_version"] = "candidate_v1"
        manifest["sha256"] = digest
        manifest["artifact_uri"] = "s3://iqa-models/candidate_v1/checkpoint.pt"
        manifest["decision_thresholds"]["model_version"] = "candidate_v1"

        manifest_path = tmp_path / "model_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        with mlflow.start_run(run_name="promoted_bundle_test") as run:
            run_id = run.info.run_id

        result = log_promoted_feature_ae_bundle(
            run_id=run_id,
            checkpoint_path=checkpoint_path,
            manifest_path=manifest_path,
            scenario_id="production_replay_natural",
            candidate_version="candidate_v1",
        )

        assert result["model_id"].startswith("m-")
        assert result["model_uri"] == (f"models:/{result['model_id']}")
        assert result["checkpoint_sha256"] == digest

        client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_tracking_uri)
        root_paths = {
            item.path for item in client.list_logged_model_artifacts(result["model_id"])
        }
        artifact_paths = {
            item.path
            for item in client.list_logged_model_artifacts(
                result["model_id"],
                "artifacts",
            )
        }

        assert "MLmodel" in root_paths
        assert {
            "artifacts/checkpoint.pt",
            "artifacts/model_manifest.json",
            "artifacts/score_contract.json",
        }.issubset(artifact_paths)

    def test_set_tags(self, mlflow_tracking_uri: str) -> None:
        """Test that logger sets traceability tags."""
        logger = MLflowRunLogger(
            run_name="test_run",
            scenario_id="test_scenario",
            tracking_uri=mlflow_tracking_uri,
        )

        logger.set_tags(
            git_commit="abc123def456",
            dataset_version="dataset_v2",
            scenario_id="test_scenario",
            manifest_version="dataset_v2_manifest_v001",
        )

        run_id = logger.end_run()
        assert run_id is not None


class TestMLflowRunLoggerIntegration:
    """Integration test: full training logging workflow."""

    def test_full_training_logging(
        self, tmp_path: Path, mlflow_tracking_uri: str
    ) -> None:
        """Test complete workflow: config → params → metrics → artifacts → tags."""
        config = FeatureAETrainingConfig(
            manifest_path=tmp_path / "manifest.json",
            image_root=tmp_path / "images",
            output_checkpoint=tmp_path / "checkpoint.pt",
            batch_size=16,
            epochs=2,
            learning_rate=5e-5,
            scenario_id="production_replay_natural",
            dataset_version="dataset_v1",
            manifest_version="dataset_v1_manifest_v001",
        )

        logger = MLflowRunLogger(
            run_name="full_training_test",
            scenario_id=config.scenario_id,
            tracking_uri=mlflow_tracking_uri,
        )

        # Log config and tags
        logger.log_config(config)
        logger.set_tags(
            git_commit="abc123",
            dataset_version=config.dataset_version,
            scenario_id=config.scenario_id,
            manifest_version=config.manifest_version,
        )

        # Log metrics for each epoch
        for epoch in range(1, 3):
            logger.log_metrics(
                {
                    "train_loss": 0.5 - (epoch * 0.1),
                    "val_loss": 0.6 - (epoch * 0.1),
                },
                step=epoch,
            )

        # Log evaluation report
        eval_report = EvaluationReport(
            model_version="v1",
            average_precision=0.95,
            recall=0.92,
            orange_rate=0.05,
            latency_ms=45.2,
            sample_count=100,
        )
        logger.log_evaluation_metrics(eval_report)

        # Log artifacts
        checkpoint_path = tmp_path / "checkpoint.pt"
        torch.save({"state_dict": {}}, checkpoint_path)
        report_path = tmp_path / "eval_report.json"
        report_path.write_text(eval_report.to_json())

        logger.log_artifacts(
            checkpoint_path=checkpoint_path, eval_report_path=report_path
        )

        # End run and verify
        run_id = logger.end_run()
        assert run_id is not None


class TestMLflowRunLoggerVerification:
    """Verify that logged fields are present in MLflow runs."""

    def test_run_contains_all_required_fields(self, mlflow_tracking_uri: str) -> None:
        """Test that MLflow run contains all required traceability fields."""
        import mlflow

        config = FeatureAETrainingConfig(
            manifest_path="manifest.json",
            image_root="images",
            output_checkpoint="checkpoint.pt",
            batch_size=16,
            epochs=10,
            learning_rate=5e-5,
            scenario_id="test_scenario",
            dataset_version="v1",
            manifest_version="v1_manifest_v001",
        )

        logger = MLflowRunLogger(
            run_name="verification_test",
            scenario_id=config.scenario_id,
            tracking_uri=mlflow_tracking_uri,
        )

        logger.log_config(config)
        logger.log_metrics({"train_loss": 0.5, "val_loss": 0.6}, step=1)
        logger.set_tags(
            git_commit="abc123",
            dataset_version="v1",
            scenario_id="test_scenario",
            manifest_version="v1_manifest_v001",
        )
        run_id = logger.end_run()

        # Verify fields in MLflow
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        run = mlflow.get_run(run_id)

        # Check params
        assert run.data.params["batch_size"] == "16"
        assert run.data.params["epochs"] == "10"
        assert run.data.params["scenario_id"] == "test_scenario"
        assert run.data.params["manifest_version"] == "v1_manifest_v001"

        # Check metrics
        assert "train_loss" in run.data.metrics
        assert "val_loss" in run.data.metrics

        # Check tags
        assert run.data.tags["git_commit"] == "abc123"
        assert run.data.tags["dataset_version"] == "v1"
        assert run.data.tags["manifest_version"] == "v1_manifest_v001"
        assert run.data.tags["scenario_id"] == "test_scenario"


class TestAirflowWrapper:
    """Test Airflow wrapper for training with MLflow logging."""

    def test_wrapper_with_mock_training(
        self, tmp_path: Path, mlflow_tracking_uri: str
    ) -> None:
        """Test that wrapper calls training and logs to MLflow."""
        config = FeatureAETrainingConfig(
            manifest_path=tmp_path / "manifest.json",
            image_root=tmp_path / "images",
            output_checkpoint=tmp_path / "checkpoint.pt",
            batch_size=16,
            epochs=1,
            learning_rate=5e-5,
            scenario_id="test_scenario",
            dataset_version="v1",
            manifest_version="v1_manifest_v001",
            run_name="test_wrapper",
        )
        config.manifest_path.write_text(
            "image_path,is_defective\nfoo.jpg,false\n", encoding="utf-8"
        )

        # Create dummy checkpoint
        config.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": {}}, config.output_checkpoint)

        # Mock the training function to avoid actual training
        with mock.patch("iqa.training.feature_ae.train_feature_ae") as mock_train:
            mock_train.return_value = {
                "model_type": "feature_ae",
                "checkpoint": str(config.output_checkpoint),
                "run_dir": str(tmp_path),
                "train_samples": 100,
                "val_samples": 20,
                "steps": 100,
                "best_epoch": 1,
                "best_loss": 0.5,
                "preprocessing_mode": "tiled_context",
            }

            # Call wrapper with tracking URI
            result = train_feature_ae_with_mlflow_logging(
                config=config,
                git_commit="abc123def456",
                tracking_uri=mlflow_tracking_uri,
            )

            # Verify training was called with correct config
            mock_train.assert_called_once_with(config)

            # Verify result is returned
            assert result["model_type"] == "feature_ae"
            assert result["checkpoint"] == str(config.output_checkpoint)
            assert result["mlflow_dataset_logged"] is True
            assert result["mlflow_training_dataset_logged"] is True
            assert result["mlflow_model_logged"] is True
