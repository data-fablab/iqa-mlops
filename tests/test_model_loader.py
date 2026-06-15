"""Tests for production model loading from MLflow and MinIO artifacts."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from iqa.inference.model_loader import LoadedModel, ProdModelLoader


class TestLoadProdModelByScenario:
    """Tracer bullet: Load prod model by scenario_id from MLflow + MinIO."""

    def test_load_prod_model_by_scenario_id(self, tmp_path: Path) -> None:
        """Loading prod model by scenario_id returns loaded model with metadata."""
        scenario_id = "production_replay_natural"
        checkpoint_path = tmp_path / "checkpoint.pt"

        checkpoint = torch.nn.Linear(10, 5)
        torch.save(checkpoint.state_dict(), checkpoint_path)

        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()
        checkpoint_artifact = artifact_dir / "checkpoint.pt"
        torch.save(checkpoint.state_dict(), checkpoint_artifact)

        with patch("iqa.inference.model_loader.registered_model_name") as mock_reg_name, \
             patch("iqa.inference.model_loader.resolve_model_artifacts") as mock_resolve, \
             patch("iqa.inference.model_loader.load_rd_feature_ae_gated") as mock_load:

            mock_reg_name.return_value = "feature_ae__production_replay_natural"
            mock_resolve.return_value = {
                "artifact_uri": str(artifact_dir),
                "stage": "prod",
                "registered_model_name": "feature_ae__production_replay_natural",
            }
            mock_model = MagicMock()
            mock_load.return_value = mock_model

            loader = ProdModelLoader(scenario_id)
            result = loader.load()

            assert isinstance(result, LoadedModel)
            assert result.scenario_id == scenario_id
            assert result.registered_model_name == "feature_ae__production_replay_natural"
            assert result.version == "unknown"
            assert result.artifact_uri == str(artifact_dir)
            assert result.model is mock_model


class TestGetProdModelReference:
    """Get prod model reference for scenario_id."""

    def test_constructs_registered_model_name_from_scenario_id(self, tmp_path: Path) -> None:
        """Registered model name is constructed from scenario_id."""
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()
        checkpoint_artifact = artifact_dir / "checkpoint.pt"
        torch.save(torch.nn.Linear(10, 5).state_dict(), checkpoint_artifact)

        with patch("iqa.inference.model_loader.registered_model_name") as mock_reg_name, \
             patch("iqa.inference.model_loader.resolve_model_artifacts") as mock_resolve, \
             patch("iqa.inference.model_loader.load_rd_feature_ae_gated") as mock_load:

            mock_reg_name.return_value = "feature_ae__production_replay_natural"
            mock_resolve.return_value = {
                "artifact_uri": str(artifact_dir),
                "stage": "prod",
            }
            mock_load.return_value = MagicMock()

            loader = ProdModelLoader("production_replay_natural")
            assert loader.scenario_id == "production_replay_natural"

            loader.load()
            mock_reg_name.assert_called_once_with("production_replay_natural")

    def test_fetch_prod_model_from_mlflow(self, tmp_path: Path) -> None:
        """Load resolves artifact from MLflow for prod stage."""
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()
        checkpoint_artifact = artifact_dir / "checkpoint.pt"
        torch.save(torch.nn.Linear(10, 5).state_dict(), checkpoint_artifact)

        with patch("iqa.inference.model_loader.registered_model_name") as mock_reg_name, \
             patch("iqa.inference.model_loader.resolve_model_artifacts") as mock_resolve, \
             patch("iqa.inference.model_loader.load_rd_feature_ae_gated") as mock_load:

            mock_reg_name.return_value = "feature_ae__production_replay_natural"
            mock_resolve.return_value = {
                "artifact_uri": str(artifact_dir),
                "stage": "prod",
            }
            mock_load.return_value = MagicMock()

            loader = ProdModelLoader("production_replay_natural")
            loader.load()

            mock_resolve.assert_called_once_with(
                "feature_ae__production_replay_natural",
                stage="prod",
                tracking_uri=None,
            )


class TestResolveAndLoadArtifact:
    """Resolve artifact URI and load checkpoint model."""

    def test_resolve_checkpoint_path_from_artifact_uri(self, tmp_path: Path) -> None:
        """Checkpoint path resolved from artifact URI."""
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()
        checkpoint_artifact = artifact_dir / "checkpoint.pt"
        torch.save(torch.nn.Linear(10, 5).state_dict(), checkpoint_artifact)

        with patch("iqa.inference.model_loader.registered_model_name"), \
             patch("iqa.inference.model_loader.resolve_model_artifacts") as mock_resolve, \
             patch("iqa.inference.model_loader.load_rd_feature_ae_gated") as mock_load:

            mock_resolve.return_value = {
                "artifact_uri": str(artifact_dir),
                "stage": "prod",
            }
            mock_load.return_value = MagicMock()

            loader = ProdModelLoader("production_replay_natural")
            loader.load()

            mock_load.assert_called_once()
            call_args = mock_load.call_args[0]
            assert str(artifact_dir / "checkpoint.pt") == str(call_args[0])

    def test_load_fails_when_checkpoint_not_found(self, tmp_path: Path) -> None:
        """Loading fails when checkpoint.pt not found in artifact directory."""
        artifact_dir = tmp_path / "empty_artifacts"
        artifact_dir.mkdir()

        with patch("iqa.inference.model_loader.registered_model_name"), \
             patch("iqa.inference.model_loader.resolve_model_artifacts") as mock_resolve, \
             patch("iqa.inference.model_loader.load_rd_feature_ae_gated"):

            mock_resolve.return_value = {
                "artifact_uri": str(artifact_dir),
                "stage": "prod",
            }

            loader = ProdModelLoader("production_replay_natural")
            with pytest.raises(FileNotFoundError):
                loader.load()

    def test_s3_uri_downloads_via_mlflow_artifacts(self, tmp_path: Path) -> None:
        """S3 URI triggers mlflow.artifacts.download_artifacts and returns local checkpoint."""
        local_dir = tmp_path / "downloaded"
        local_dir.mkdir()
        (local_dir / "checkpoint.pt").touch()

        with patch("iqa.inference.model_loader.registered_model_name"), \
             patch("iqa.inference.model_loader.resolve_model_artifacts") as mock_resolve, \
             patch("iqa.inference.model_loader.load_rd_feature_ae_gated") as mock_load, \
             patch("mlflow.artifacts.download_artifacts", return_value=str(local_dir)) as mock_dl:

            mock_resolve.return_value = {
                "artifact_uri": "s3://mlflow-artifacts/run123/artifacts",
                "stage": "prod",
            }
            mock_load.return_value = MagicMock()

            loader = ProdModelLoader("production_replay_natural")
            loader.load()

            mock_dl.assert_called_once_with(artifact_uri="s3://mlflow-artifacts/run123/artifacts")
            mock_load.assert_called_once_with(local_dir / "checkpoint.pt")

    def test_s3_uri_raises_file_not_found_when_checkpoint_absent(self, tmp_path: Path) -> None:
        """S3 download succeeds but checkpoint.pt absent raises FileNotFoundError."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with patch("iqa.inference.model_loader.registered_model_name"), \
             patch("iqa.inference.model_loader.resolve_model_artifacts") as mock_resolve, \
             patch("mlflow.artifacts.download_artifacts", return_value=str(empty_dir)):

            mock_resolve.return_value = {
                "artifact_uri": "s3://mlflow-artifacts/run123/artifacts",
                "stage": "prod",
            }

            loader = ProdModelLoader("production_replay_natural")
            with pytest.raises(FileNotFoundError):
                loader.load()


class TestModelVersionTracking:
    """Track which model version is loaded."""

    def test_track_loaded_model_version(self, tmp_path: Path) -> None:
        """Loaded model version is extracted from artifact URI."""
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()
        checkpoint_artifact = artifact_dir / "checkpoint.pt"
        torch.save(torch.nn.Linear(10, 5).state_dict(), checkpoint_artifact)

        with patch("iqa.inference.model_loader.registered_model_name"), \
             patch("iqa.inference.model_loader.resolve_model_artifacts") as mock_resolve, \
             patch("iqa.inference.model_loader.load_rd_feature_ae_gated") as mock_load:

            run_id = "abc123def456"
            mlruns_dir = tmp_path / "mlruns" / "0" / run_id / "artifacts"
            mlruns_dir.mkdir(parents=True)
            checkpoint_mlruns = mlruns_dir / "checkpoint.pt"
            torch.save(torch.nn.Linear(10, 5).state_dict(), checkpoint_mlruns)

            mock_resolve.return_value = {
                "artifact_uri": str(mlruns_dir),
                "stage": "prod",
            }
            mock_load.return_value = MagicMock()

            loader = ProdModelLoader("production_replay_natural")
            result = loader.load()

            assert result.version == run_id

    def test_version_unknown_when_uri_format_unexpected(self, tmp_path: Path) -> None:
        """Version is 'unknown' when URI doesn't contain recognizable run_id."""
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()
        checkpoint_artifact = artifact_dir / "checkpoint.pt"
        torch.save(torch.nn.Linear(10, 5).state_dict(), checkpoint_artifact)

        with patch("iqa.inference.model_loader.registered_model_name"), \
             patch("iqa.inference.model_loader.resolve_model_artifacts") as mock_resolve, \
             patch("iqa.inference.model_loader.load_rd_feature_ae_gated") as mock_load:

            mock_resolve.return_value = {
                "artifact_uri": str(artifact_dir),
                "stage": "prod",
            }
            mock_load.return_value = MagicMock()

            loader = ProdModelLoader("production_replay_natural")
            result = loader.load()

            assert result.version == "unknown"


class TestReloadAfterPromotion:
    """Support reloading model after promotion."""

    def test_reload_refreshes_model_from_mlflow(self, tmp_path: Path) -> None:
        """Reload fetches fresh model from MLflow."""
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()
        checkpoint_artifact = artifact_dir / "checkpoint.pt"
        torch.save(torch.nn.Linear(10, 5).state_dict(), checkpoint_artifact)

        with patch("iqa.inference.model_loader.registered_model_name"), \
             patch("iqa.inference.model_loader.resolve_model_artifacts") as mock_resolve, \
             patch("iqa.inference.model_loader.load_rd_feature_ae_gated") as mock_load:

            mock_resolve.return_value = {
                "artifact_uri": str(artifact_dir),
                "stage": "prod",
            }
            mock_model_1 = MagicMock(name="model_v1")
            mock_model_2 = MagicMock(name="model_v2")
            mock_load.side_effect = [mock_model_1, mock_model_2]

            loader = ProdModelLoader("production_replay_natural")
            result_1 = loader.load()
            result_2 = loader.reload()

            assert result_1.model is mock_model_1
            assert result_2.model is mock_model_2
            assert mock_load.call_count == 2

    def test_current_returns_loaded_model_without_reload(self, tmp_path: Path) -> None:
        """Current returns loaded model without reloading."""
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()
        checkpoint_artifact = artifact_dir / "checkpoint.pt"
        torch.save(torch.nn.Linear(10, 5).state_dict(), checkpoint_artifact)

        with patch("iqa.inference.model_loader.registered_model_name"), \
             patch("iqa.inference.model_loader.resolve_model_artifacts") as mock_resolve, \
             patch("iqa.inference.model_loader.load_rd_feature_ae_gated") as mock_load:

            mock_resolve.return_value = {
                "artifact_uri": str(artifact_dir),
                "stage": "prod",
            }
            mock_load.return_value = MagicMock()

            loader = ProdModelLoader("production_replay_natural")
            assert loader.current() is None

            loaded = loader.load()
            assert loader.current() is loaded
            assert mock_load.call_count == 1

            current_again = loader.current()
            assert current_again is loaded
            assert mock_load.call_count == 1

    def test_current_returns_none_before_load(self) -> None:
        """Current returns None before any model loaded."""
        loader = ProdModelLoader("production_replay_natural")
        assert loader.current() is None


class TestTrackingUri:
    """Support custom MLflow tracking URI."""

    def test_pass_tracking_uri_to_resolve_artifacts(self, tmp_path: Path) -> None:
        """Tracking URI is passed to resolve_model_artifacts."""
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()
        checkpoint_artifact = artifact_dir / "checkpoint.pt"
        torch.save(torch.nn.Linear(10, 5).state_dict(), checkpoint_artifact)

        tracking_uri = "http://mlflow-server:5000"

        with patch("iqa.inference.model_loader.registered_model_name") as mock_reg_name, \
             patch("iqa.inference.model_loader.resolve_model_artifacts") as mock_resolve, \
             patch("iqa.inference.model_loader.load_rd_feature_ae_gated") as mock_load:

            model_name = "feature_ae__production_replay_natural"
            mock_reg_name.return_value = model_name
            mock_resolve.return_value = {
                "artifact_uri": str(artifact_dir),
                "stage": "prod",
            }
            mock_load.return_value = MagicMock()

            loader = ProdModelLoader("production_replay_natural", tracking_uri=tracking_uri)
            loader.load()

            mock_resolve.assert_called_once_with(
                model_name,
                stage="prod",
                tracking_uri=tracking_uri,
            )


__all__ = [
    "TestLoadProdModelByScenario",
    "TestGetProdModelReference",
    "TestResolveAndLoadArtifact",
    "TestModelVersionTracking",
    "TestReloadAfterPromotion",
    "TestTrackingUri",
]
