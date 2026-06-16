"""Feature AE candidate model with train/eval/save/load/predict interface."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from iqa.models.feature_ae import (
    DEFAULT_FEATURE_LAYERS,
    FEATURE_AE_MODEL_TYPE,
    ReverseDistillationGatedDualContextResNet18,
    load_rd_feature_ae_gated,
    normalize_feature_layers,
)
# NOTE: iqa.training.* imports are deferred (TYPE_CHECKING + lazy in methods) to
# break the iqa.training.feature_ae <-> iqa.models.feature_ae_candidate import cycle.
if TYPE_CHECKING:
    from iqa.inference.feature_ae import FeatureAEPrediction
    from iqa.training.feature_ae import FeatureAETrainingConfig
    from iqa.training.feature_ae_evaluation import FeatureAEEvaluationConfig
else:
    FeatureAEPrediction = None


def _get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _save_candidate_metadata(checkpoint_path: Path, config: FeatureAETrainingConfig) -> None:
    """Save metadata alongside checkpoint for traceability."""
    metadata = {
        "candidate_version": config.candidate_version,
        "dataset_version": config.dataset_version,
        "git_commit": _get_git_commit(),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "scenario_id": config.scenario_id,
        "model_type": FEATURE_AE_MODEL_TYPE,
    }
    metadata_path = checkpoint_path.parent / f"{checkpoint_path.stem}.metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))


class FeatureAECandidate:
    """Candidate Feature AE model with standardized interface."""

    def __init__(self, model: ReverseDistillationGatedDualContextResNet18) -> None:
        self.model = model
        self.model.eval()

    @classmethod
    def train(cls, config: FeatureAETrainingConfig) -> FeatureAECandidate:
        """Train a new Feature AE candidate from configuration.

        Args:
            config: Training configuration with dataset paths, hyperparams, etc.

        Returns:
            FeatureAECandidate with trained model.

        Saves:
            Checkpoint at config.output_checkpoint.
            Metadata with version/git commit at checkpoint_stem.metadata.json.
        """
        from iqa.training.feature_ae import train_feature_ae

        results = train_feature_ae(config)
        checkpoint_path = Path(results.get("checkpoint_path", config.output_checkpoint))

        # Save metadata for traceability
        _save_candidate_metadata(checkpoint_path, config)

        model = load_rd_feature_ae_gated(checkpoint_path)
        return cls(model=model)

    @classmethod
    def load(cls, checkpoint_path: Path | str) -> FeatureAECandidate:
        """Load a Feature AE candidate from checkpoint.

        Args:
            checkpoint_path: Path to saved checkpoint.

        Returns:
            FeatureAECandidate with loaded model.
        """
        checkpoint_path = Path(checkpoint_path)
        model = load_rd_feature_ae_gated(checkpoint_path)
        return cls(model=model)

    def predict(self, image_path: str | Path, **kwargs) -> dict[str, Any]:
        """Predict on image, conformant to model contract.

        Args:
            image_path: Path to image.
            **kwargs: Additional kwargs for predict_feature_ae_image (thresholds, device, etc).

        Returns:
            FeatureAEPrediction with score, status, latency, etc.
        """
        from iqa.datasets import FEATURE_AE_CONTEXT_SIZE, FEATURE_AE_TILE_SIZE, load_image_tensor
        from iqa.inference.feature_ae import FeatureAEPrediction
        from iqa.inference.helpers import compute_status, measure_inference_time
        from iqa.models.feature_ae import feature_anomaly_map, ResNetTeacherFeatures

        image_size = kwargs.pop('image_size', FEATURE_AE_TILE_SIZE)
        context_size = kwargs.pop('context_size', FEATURE_AE_CONTEXT_SIZE)
        preprocessing_mode = kwargs.pop('preprocessing_mode', 'tiled_context')
        threshold_orange = kwargs.pop('threshold_orange', 0.02)
        threshold_red = kwargs.pop('threshold_red', 0.05)
        device = kwargs.pop('device', 'cpu')
        pretrained_teacher = kwargs.pop('pretrained_teacher', False)
        layers = kwargs.pop('layers', DEFAULT_FEATURE_LAYERS)

        layers = normalize_feature_layers(layers)
        torch_device = torch.device(device)
        image = load_image_tensor(image_path, image_size=image_size).unsqueeze(0).to(torch_device)
        context_image_size = context_size if preprocessing_mode == 'tiled_context' else image_size
        context_image = load_image_tensor(image_path, image_size=context_image_size).unsqueeze(0).to(torch_device)

        self.model.to(torch_device)
        teacher = ResNetTeacherFeatures(layers=layers, pretrained=pretrained_teacher).to(torch_device)
        teacher.eval()

        with torch.no_grad():
            with measure_inference_time() as timing:
                teacher_features = teacher(image)
                reconstructed = self.model(image, context_images=context_image)
                anomaly_map = feature_anomaly_map(teacher_features, reconstructed)

        score = float(anomaly_map.mean().detach().cpu())
        status_lower = compute_status(score, threshold_orange=threshold_orange, threshold_red=threshold_red)
        # Convert to contract format: red→Rouge, orange→Orange, green→Vert
        status_map = {"red": "Rouge", "orange": "Orange", "green": "Vert"}
        status = status_map.get(status_lower, status_lower)

        return FeatureAEPrediction(
            image_path=str(image_path),
            model_type=FEATURE_AE_MODEL_TYPE,
            score=score,
            status=status,
            threshold_orange=float(threshold_orange),
            threshold_red=float(threshold_red),
            latency_ms=timing['elapsed_ms'],
            roi_status=None,
            heatmap_uri=None,
        )

    def save(self, output_checkpoint: Path) -> dict[str, str]:
        """Save checkpoint to disk.

        Args:
            output_checkpoint: Path where checkpoint will be saved.

        Returns:
            Metadata dict with checkpoint path and info.
        """
        output_checkpoint = Path(output_checkpoint)
        output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), output_checkpoint)
        return {"checkpoint_path": str(output_checkpoint), "model_type": "feature_ae"}

    def eval(self, config: FeatureAEEvaluationConfig) -> dict[str, Any]:
        """Evaluate on a validation set and compute metrics.

        Args:
            config: Evaluation configuration.

        Returns:
            Metrics dict with AP, recall, latency, etc.
        """
        from iqa.training.feature_ae_evaluation import evaluate_feature_ae_checkpoint

        metrics = evaluate_feature_ae_checkpoint(config)
        return metrics


__all__ = ["FeatureAECandidate"]
