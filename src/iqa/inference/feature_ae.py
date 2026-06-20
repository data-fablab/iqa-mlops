"""Feature-AE image prediction for the reproducible IQA source path."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from iqa.inference.helpers import compute_status, measure_inference_time
from iqa.models.feature_ae import (
    CHAMPION_FEATURE_AE_CONTRACT,
    FeatureAEChampionContract,
    DEFAULT_FEATURE_LAYERS,
    FEATURE_AE_MODEL_TYPE,
    ResNetTeacherFeatures,
    apply_champion_roi,
    fuse_numpy_layer_maps,
    load_roi_probability_map,
    load_rd_feature_ae_gated,
    normalize_feature_layers,
    reconstruct_tiled_feature_maps,
    score_numpy_map_topk,
    smooth_numpy_score_map,
)
from iqa.training.feature_ae_contracts import CANONICAL_FEATURE_AE_PREPROCESSING


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
    threshold_source: str = "explicit_or_default"
    score_contract_version: str = CHAMPION_FEATURE_AE_CONTRACT.version

    def to_dict(self) -> dict[str, float | str | None]:
        return asdict(self)


@dataclass(frozen=True)
class FeatureAEScoreMaps:
    layer_maps: dict[str, np.ndarray]
    fused_map: np.ndarray
    score_map: np.ndarray
    roi_probability: np.ndarray | None
    score: float
    contract: FeatureAEChampionContract


def predict_feature_ae_image(
    image_path: str | Path,
    checkpoint_path: str | Path,
    *,
    image_size: int = CANONICAL_FEATURE_AE_PREPROCESSING.image_size,
    context_size: int = CANONICAL_FEATURE_AE_PREPROCESSING.context_size,
    preprocessing_mode: str = CANONICAL_FEATURE_AE_PREPROCESSING.preprocessing_mode,
    threshold_orange: float = 0.02,
    threshold_red: float = 0.05,
    threshold_source: str = "explicit_or_default",
    roi_mask_path: str | Path | None = None,
    roi_probability_path: str | Path | None = None,
    score_smoothing: str = CANONICAL_FEATURE_AE_PREPROCESSING.score_smoothing,
    score_image: str = CANONICAL_FEATURE_AE_PREPROCESSING.score_image,
    topk_fraction: float = CANONICAL_FEATURE_AE_PREPROCESSING.topk_fraction,
    heatmap_output_path: str | Path | None = None,
    heatmap_uri: str | None = None,
    device: str = "cpu",
    pretrained_teacher: bool = True,
    layers: tuple[str, ...] = DEFAULT_FEATURE_LAYERS,
    champion_contract: FeatureAEChampionContract = CHAMPION_FEATURE_AE_CONTRACT,
) -> FeatureAEPrediction:
    with measure_inference_time() as timing:
        maps = compute_feature_ae_score_maps(
            image_path,
            checkpoint_path,
            image_size=image_size,
            context_size=context_size,
            preprocessing_mode=preprocessing_mode,
            roi_mask_path=roi_mask_path,
            roi_probability_path=roi_probability_path,
            score_smoothing=score_smoothing,
            score_image=score_image,
            topk_fraction=topk_fraction,
            device=device,
            pretrained_teacher=pretrained_teacher,
            layers=layers,
            champion_contract=champion_contract,
        )
    score_map_array = maps.score_map
    if heatmap_output_path is not None:
        save_feature_ae_heatmap_overlay(
            image_path,
            torch.from_numpy(score_map_array),
            heatmap_output_path,
            roi_mask_path=roi_mask_path,
            threshold_orange=threshold_orange,
            threshold_red=threshold_red,
        )
    score = maps.score
    status = compute_status(score, threshold_orange=threshold_orange, threshold_red=threshold_red)
    return FeatureAEPrediction(
        image_path=str(image_path),
        model_type=FEATURE_AE_MODEL_TYPE,
        score=score,
        status=status,
        threshold_orange=float(threshold_orange),
        threshold_red=float(threshold_red),
        latency_ms=timing["elapsed_ms"],
        roi_status="roi_scored" if roi_mask_path is not None else None,
        heatmap_uri=heatmap_uri or (str(heatmap_output_path) if heatmap_output_path is not None else None),
        threshold_source=threshold_source,
        score_contract_version=maps.contract.version,
    )


def compute_feature_ae_score_maps(
    image_path: str | Path,
    checkpoint_path: str | Path,
    *,
    image_size: int = CANONICAL_FEATURE_AE_PREPROCESSING.image_size,
    context_size: int = CANONICAL_FEATURE_AE_PREPROCESSING.context_size,
    preprocessing_mode: str = CANONICAL_FEATURE_AE_PREPROCESSING.preprocessing_mode,
    roi_mask_path: str | Path | None = None,
    roi_probability_path: str | Path | None = None,
    score_smoothing: str = CANONICAL_FEATURE_AE_PREPROCESSING.score_smoothing,
    score_image: str = CANONICAL_FEATURE_AE_PREPROCESSING.score_image,
    topk_fraction: float = CANONICAL_FEATURE_AE_PREPROCESSING.topk_fraction,
    device: str = "cpu",
    pretrained_teacher: bool = True,
    layers: tuple[str, ...] = DEFAULT_FEATURE_LAYERS,
    champion_contract: FeatureAEChampionContract = CHAMPION_FEATURE_AE_CONTRACT,
) -> FeatureAEScoreMaps:
    layers = normalize_feature_layers(layers)
    torch_device = torch.device(device)
    model = load_rd_feature_ae_gated(checkpoint_path, layers=layers, map_location=torch_device).to(torch_device)
    teacher = ResNetTeacherFeatures(layers=layers, pretrained=pretrained_teacher).to(torch_device)
    model.eval()
    teacher.eval()
    contract = FeatureAEChampionContract(
        version=champion_contract.version,
        teacher_weights=champion_contract.teacher_weights,
        tile_size=int(image_size),
        context_size=int(context_size if preprocessing_mode == "tiled_context" else image_size),
        tile_stride=champion_contract.tile_stride,
        layers=layers,
        layer_weights=champion_contract.normalized_layer_weights(),
        score_smoothing=score_smoothing,
        roi_mode=champion_contract.roi_mode,
        roi_threshold=champion_contract.roi_threshold,
        score_image=score_image,
        topk_fraction=topk_fraction,
    )
    layer_maps = reconstruct_tiled_feature_maps(
        image_path=image_path,
        model=model,
        teacher=teacher,
        device=torch_device,
        contract=contract,
    )
    fused_map = fuse_numpy_layer_maps(layer_maps, layer_weights=contract.normalized_layer_weights())
    smoothed_map = smooth_numpy_score_map(fused_map, score_smoothing)
    roi_source = roi_probability_path or roi_mask_path
    roi_probability = load_roi_probability_map(
        roi_source,
        target_shape=smoothed_map.shape,
        threshold=contract.roi_threshold,
    )
    score_map_array = apply_champion_roi(
        smoothed_map,
        roi_probability=roi_probability,
        roi_mode=contract.roi_mode,
    )
    score = score_numpy_map_topk(
        score_map_array,
        roi_probability=roi_probability,
        score_image=score_image,
        topk_fraction=topk_fraction,
    )
    return FeatureAEScoreMaps(
        layer_maps=layer_maps,
        fused_map=fused_map,
        score_map=score_map_array,
        roi_probability=roi_probability,
        score=score,
        contract=contract,
    )


def score_feature_ae_map(
    score_map: torch.Tensor,
    *,
    roi_mask_path: str | Path | None = None,
    score_smoothing: str = CANONICAL_FEATURE_AE_PREPROCESSING.score_smoothing,
    score_image: str = CANONICAL_FEATURE_AE_PREPROCESSING.score_image,
    topk_fraction: float = CANONICAL_FEATURE_AE_PREPROCESSING.topk_fraction,
) -> float:
    score_map = prepare_feature_ae_score_map(
        score_map,
        roi_mask_path=roi_mask_path,
        score_smoothing=score_smoothing,
    )
    valid_mask = load_roi_mask(roi_mask_path, target_shape=score_map.shape) if roi_mask_path is not None else None
    return score_image_map(score_map, score_image=score_image, topk_fraction=topk_fraction, valid_mask=valid_mask)


def prepare_feature_ae_score_map(
    score_map: torch.Tensor,
    *,
    roi_mask_path: str | Path | None = None,
    score_smoothing: str = CANONICAL_FEATURE_AE_PREPROCESSING.score_smoothing,
) -> torch.Tensor:
    score_map = score_map.to(dtype=torch.float32)
    if score_map.ndim != 2:
        raise ValueError(f"Feature-AE score map must be 2D, got shape={tuple(score_map.shape)}")
    score_map = smooth_score_map(score_map, score_smoothing)
    valid_mask = load_roi_mask(roi_mask_path, target_shape=score_map.shape) if roi_mask_path is not None else None
    if valid_mask is not None:
        score_map = score_map.masked_fill(~valid_mask, 0.0)
    return score_map


def smooth_score_map(score_map: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return score_map
    if mode == "median3":
        padded = F.pad(score_map.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="replicate")
        windows = F.unfold(padded, kernel_size=3).squeeze(0)
        return windows.median(dim=0).values.reshape_as(score_map)
    raise ValueError(f"Unsupported Feature-AE score_smoothing: {mode!r}")


def score_image_map(
    score_map: torch.Tensor,
    *,
    score_image: str,
    topk_fraction: float,
    valid_mask: torch.Tensor | None = None,
) -> float:
    values = score_map[valid_mask] if valid_mask is not None else score_map.flatten()
    if values.numel() == 0:
        values = score_map.flatten()
    if score_image == "max":
        return float(values.max().item())
    if score_image == "mean":
        return float(values.mean().item())
    if score_image != "topk_mean":
        raise ValueError(f"Unsupported Feature-AE score_image: {score_image!r}")
    k = max(1, int(np.ceil(values.numel() * float(topk_fraction))))
    return float(torch.topk(values, min(k, values.numel())).values.mean().item())


def load_roi_mask(roi_mask_path: str | Path, *, target_shape: torch.Size | tuple[int, int]) -> torch.Tensor:
    mask = Image.open(roi_mask_path).convert("L")
    height, width = int(target_shape[0]), int(target_shape[1])
    if mask.size != (width, height):
        mask = mask.resize((width, height), Image.Resampling.NEAREST)
    array = np.asarray(mask)
    return torch.from_numpy(array > 0)


def save_feature_ae_heatmap_overlay(
    image_path: str | Path,
    score_map: torch.Tensor,
    output_path: str | Path,
    *,
    roi_mask_path: str | Path | None = None,
    threshold_orange: float | None = None,
    threshold_red: float | None = None,
    display_low_percentile: float = 85.0,
    display_high_percentile: float = 99.8,
    display_gamma: float = 1.4,
    display_threshold: float = 0.60,
    overlay_alpha: float = 0.72,
) -> None:
    """Save an operator anomaly overlay PNG.

    Decision thresholds are kept in the signature for traceability, but the
    visual overlay uses the champion preview contract: ROI-local percentile
    normalization plus a display threshold. This keeps low calibrated decision
    thresholds from painting the whole ROI red.
    """

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    score_array = score_map.detach().cpu().numpy().astype(np.float32)
    base = Image.open(image_path).convert("RGB")
    roi_mask_array: np.ndarray | None = None
    if roi_mask_path is not None:
        roi_mask = Image.open(roi_mask_path).convert("L")
        if roi_mask.size != (score_array.shape[1], score_array.shape[0]):
            roi_mask = roi_mask.resize((score_array.shape[1], score_array.shape[0]), Image.Resampling.NEAREST)
        roi_mask_array = np.asarray(roi_mask, dtype=np.float32) > 0
    normalized = normalize_feature_ae_display_map(
        score_array,
        roi_mask=roi_mask_array,
        low_percentile=display_low_percentile,
        high_percentile=display_high_percentile,
        gamma=display_gamma,
        display_threshold=display_threshold,
    )
    if base.size != (score_array.shape[1], score_array.shape[0]):
        alpha_image = Image.fromarray((normalized * 255.0).astype(np.uint8), mode="L")
        alpha_image = alpha_image.resize(base.size, Image.Resampling.BILINEAR)
        normalized = np.asarray(alpha_image, dtype=np.float32) / 255.0
    base_array = np.asarray(base, dtype=np.float32)
    heat = np.zeros_like(base_array)
    heat[..., 0] = 255.0
    heat[..., 1] = np.clip(np.sqrt(normalized) * 199.0, 0.0, 199.0)
    heat[..., 2] = 5.0
    alpha = (normalized[..., None] * float(overlay_alpha)).astype(np.float32)
    overlay = (base_array * (1.0 - alpha) + heat * alpha).clip(0, 255).astype(np.uint8)
    Image.fromarray(overlay, mode="RGB").save(output)


def normalize_feature_ae_display_map(
    score_map: np.ndarray,
    *,
    roi_mask: np.ndarray | None = None,
    low_percentile: float = 85.0,
    high_percentile: float = 99.8,
    gamma: float = 1.4,
    display_threshold: float = 0.60,
) -> np.ndarray:
    """Normalize a Feature-AE map for display without changing decision scores."""

    score = np.asarray(score_map, dtype=np.float32)
    finite = np.isfinite(score)
    if roi_mask is not None:
        valid = finite & (np.asarray(roi_mask) > 0)
    else:
        valid = finite
    values = score[valid]
    if values.size == 0:
        return np.zeros_like(score, dtype=np.float32)
    low = float(np.percentile(values, float(low_percentile))) if float(low_percentile) > 0 else 0.0
    high = float(np.percentile(values, float(high_percentile)))
    if high <= low:
        high = low + 1e-6
    normalized = np.clip((score - low) / (high - low), 0.0, 1.0)
    if float(gamma) > 0.0 and float(gamma) != 1.0:
        normalized = np.power(normalized, float(gamma))
    if roi_mask is not None:
        normalized = np.where(np.asarray(roi_mask) > 0, normalized, 0.0)
    normalized[normalized < float(display_threshold)] = 0.0
    return normalized.astype(np.float32, copy=False)


__all__ = [
    "FeatureAEPrediction",
    "FeatureAEScoreMaps",
    "compute_feature_ae_score_maps",
    "load_roi_mask",
    "normalize_feature_ae_display_map",
    "prepare_feature_ae_score_map",
    "predict_feature_ae_image",
    "save_feature_ae_heatmap_overlay",
    "score_feature_ae_map",
    "score_image_map",
    "smooth_score_map",
]
