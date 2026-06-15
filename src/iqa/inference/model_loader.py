"""Load production models by scenario_id from MLflow and MinIO artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from iqa.models.feature_ae import load_rd_feature_ae_gated
from iqa.promotion import resolve_model_artifacts
from iqa.registry.mlflow_registry import registered_model_name


@dataclass(frozen=True)
class LoadedModel:
    """Metadata for a loaded production model."""

    scenario_id: str
    registered_model_name: str
    version: str
    artifact_uri: str
    model: Any


class ProdModelLoader:
    """Load and manage production models by scenario_id from MLflow and MinIO."""

    def __init__(self, scenario_id: str, *, tracking_uri: str | None = None):
        """Initialize loader for a scenario.

        Args:
            scenario_id: Scenario identifier (e.g., "production_replay_natural")
            tracking_uri: MLflow tracking URI (optional)
        """
        self.scenario_id = scenario_id
        self.tracking_uri = tracking_uri
        self._loaded_model: LoadedModel | None = None

    def load(self) -> LoadedModel:
        """Load production model for this scenario from MLflow and MinIO.

        Returns:
            LoadedModel with model instance and metadata (version, artifact_uri)

        Raises:
            ValueError: If scenario_id is invalid or model not found
            ImportError: If required dependencies unavailable
        """
        model_name = registered_model_name(self.scenario_id)

        artifact_info = resolve_model_artifacts(
            model_name,
            stage="prod",
            tracking_uri=self.tracking_uri,
        )

        artifact_uri = artifact_info["artifact_uri"]
        version = self._extract_version_from_uri(artifact_uri)

        checkpoint_path = self._resolve_checkpoint_path(artifact_uri)
        model = load_rd_feature_ae_gated(checkpoint_path)

        self._loaded_model = LoadedModel(
            scenario_id=self.scenario_id,
            registered_model_name=model_name,
            version=version,
            artifact_uri=artifact_uri,
            model=model,
        )
        return self._loaded_model

    def reload(self) -> LoadedModel:
        """Reload model after promotion (fetch fresh from MLflow).

        Returns:
            LoadedModel with updated model and metadata

        Raises:
            ValueError: If model not found
            ImportError: If required dependencies unavailable
        """
        return self.load()

    def current(self) -> LoadedModel | None:
        """Get the currently loaded model without reloading.

        Returns:
            LoadedModel if one is loaded, None otherwise
        """
        return self._loaded_model

    @staticmethod
    def _extract_version_from_uri(artifact_uri: str) -> str:
        """Extract MLflow version from artifact URI.

        Args:
            artifact_uri: S3 URI or local path with version info

        Returns:
            Version string (MLflow run_id)
        """
        # Handle MLflow local path format: .../mlruns/{exp_id}/{run_id}/artifacts
        if "mlruns" in artifact_uri:
            parts = artifact_uri.split("mlruns/")
            if len(parts) > 1:
                remainder = parts[1].split("/")
                if len(remainder) >= 2:
                    return remainder[1]

        if "runs/" in artifact_uri:
            parts = artifact_uri.split("runs/")
            if len(parts) > 1:
                run_id = parts[1].split("/")[0]
                return run_id
        return "unknown"

    @staticmethod
    def _resolve_checkpoint_path(artifact_uri: str) -> Path:
        """Resolve artifact URI to local checkpoint path.

        For now, this is a placeholder that assumes the URI is already a local path
        (as returned by local MLflow setup). In production with S3, this would
        download from MinIO.

        Args:
            artifact_uri: S3 URI or local path

        Returns:
            Path to checkpoint file
        """
        if artifact_uri.startswith("s3://"):
            import mlflow.artifacts
            local_dir = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri)
            checkpoint = Path(local_dir) / "checkpoint.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(f"Checkpoint not found at {checkpoint}")
            return checkpoint

        uri_path = Path(artifact_uri)
        checkpoint = uri_path / "checkpoint.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint}")
        return checkpoint


__all__ = ["LoadedModel", "ProdModelLoader"]
