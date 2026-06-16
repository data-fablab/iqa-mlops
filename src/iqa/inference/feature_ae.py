"""Feature-AE image prediction for the reproducible IQA source path."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from iqa.datasets import FEATURE_AE_CONTEXT_SIZE, FEATURE_AE_TILE_SIZE, load_image_tensor
from iqa.inference.helpers import compute_status, measure_inference_time
from iqa.models.feature_ae import (
    DEFAULT_FEATURE_LAYERS,
    FEATURE_AE_MODEL_TYPE,
    ResNetTeacherFeatures,
    feature_anomaly_map,
    load_rd_feature_ae_gated,
    normalize_feature_layers,
)


@dataclass(frozen=True)
class FeatureAEPrediction:
    image_path: str
    model_type: str
    score: float
    status: str
    threshold_orange: float
    threshold_red: float
    latency_ms: float
    roi_status: str | None = None
    heatmap_uri: str | None = None

    def to_dict(self) -> dict[str, float | str | None]:
        return asdict(self)


def predict_feature_ae_image(
    image_path: str | Path,
    checkpoint_path: str | Path,
    *,
    image_size: int = FEATURE_AE_TILE_SIZE,
    context_size: int = FEATURE_AE_CONTEXT_SIZE,
    preprocessing_mode: str = "tiled_context",
    threshold_orange: float = 0.02,
    threshold_red: float = 0.05,
    device: str = "cpu",
    pretrained_teacher: bool = False,
    layers: tuple[str, ...] = DEFAULT_FEATURE_LAYERS,
) -> FeatureAEPrediction:
    layers = normalize_feature_layers(layers)
    torch_device = torch.device(device)
    image = load_image_tensor(image_path, image_size=image_size).unsqueeze(0).to(torch_device)
    context_image_size = context_size if preprocessing_mode == "tiled_context" else image_size
    context_image = load_image_tensor(image_path, image_size=context_image_size).unsqueeze(0).to(torch_device)

    model = load_rd_feature_ae_gated(checkpoint_path, layers=layers, map_location=torch_device).to(torch_device)
    teacher = ResNetTeacherFeatures(layers=layers, pretrained=pretrained_teacher).to(torch_device)
    model.eval()
    teacher.eval()

    with torch.no_grad():
        with measure_inference_time() as timing:
            teacher_features = teacher(image)
            reconstructed = model(image, context_images=context_image)
            anomaly_map = feature_anomaly_map(teacher_features, reconstructed)

    score = float(anomaly_map.mean().detach().cpu())
    status = compute_status(score, threshold_orange=threshold_orange, threshold_red=threshold_red)
    return FeatureAEPrediction(
        image_path=str(image_path),
        model_type=FEATURE_AE_MODEL_TYPE,
        score=score,
        status=status,
        threshold_orange=float(threshold_orange),
        threshold_red=float(threshold_red),
        latency_ms=timing["elapsed_ms"],
        roi_status=None,
        heatmap_uri=None,
    )


__all__ = ["FeatureAEPrediction", "predict_feature_ae_image"]
