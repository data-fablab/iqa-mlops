"""Canonical Feature-AE preprocessing and selection contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from iqa.datasets import FEATURE_AE_CONTEXT_SIZE, FEATURE_AE_TILE_SIZE

FEATURE_AE_PREPROCESSING_CONTRACT_VERSION = "feature_ae_champion_v001"
FEATURE_AE_BUSINESS_METRIC_PRIORITY = (
    "pixel_aupimo_1e-5_1e-3",
    "pixel_ap",
    "image_ap",
    "image_auroc",
)
FEATURE_AE_CHAMPION_LAYER_WEIGHTS = {"layer2": 0.65, "layer3": 0.35}
FEATURE_AE_CHAMPION_TEACHER_WEIGHTS = "IMAGENET1K_V1"
FEATURE_AE_CHAMPION_ROI_MODE = "soft_map"
FEATURE_AE_REFERENCE_SCORE_MODE = "sqrt_l2_plus_cosine"
FEATURE_AE_REFERENCE_LAYER_NORMALIZATION = "good_p99"


@dataclass(frozen=True)
class FeatureAEPreprocessingContract:
    version: str = FEATURE_AE_PREPROCESSING_CONTRACT_VERSION
    preprocessing_mode: str = "tiled_context"
    image_size: int = FEATURE_AE_TILE_SIZE
    context_size: int = FEATURE_AE_CONTEXT_SIZE
    tile_stride: int = FEATURE_AE_TILE_SIZE
    normalization: str = "imagenet"
    tile_train_sampling: str = "all"
    teacher_weights: str = FEATURE_AE_CHAMPION_TEACHER_WEIGHTS
    layer_weights: dict[str, float] | None = None
    roi_mode: str = FEATURE_AE_CHAMPION_ROI_MODE
    roi_threshold: float = 0.50
    min_roi_ratio: float = 0.03
    score_region: str = "functional_surface_prediction"
    score_smoothing: str = "median3"
    score_image: str = "topk_mean"
    topk_fraction: float = 0.005
    augmentation_profile: str = "none"
    layer_score_mode: str = FEATURE_AE_REFERENCE_SCORE_MODE
    layer_normalization: str = FEATURE_AE_REFERENCE_LAYER_NORMALIZATION
    layer_normalization_stats: dict[str, float] | None = None
    cosine_weight: float = 0.5


CANONICAL_FEATURE_AE_PREPROCESSING = FeatureAEPreprocessingContract(
    layer_weights=FEATURE_AE_CHAMPION_LAYER_WEIGHTS.copy()
)


def canonical_feature_ae_preprocessing_dict() -> dict[str, Any]:
    return asdict(CANONICAL_FEATURE_AE_PREPROCESSING)


def assert_canonical_feature_ae_preprocessing(
    *,
    preprocessing_mode: str,
    image_size: int,
    context_size: int,
    tile_stride: int,
    tile_train_sampling: str,
    roi_threshold: float,
    min_roi_ratio: float,
    score_region: str | None = None,
    score_smoothing: str | None = None,
    score_image: str | None = None,
    topk_fraction: float | None = None,
    augmentation_profile: str = "none",
    allow_noncanonical_preprocessing: bool = False,
) -> None:
    """Reject preprocessing drift for comparable Feature-AE candidates."""
    if allow_noncanonical_preprocessing:
        return
    expected = CANONICAL_FEATURE_AE_PREPROCESSING
    observed: dict[str, object] = {
        "preprocessing_mode": preprocessing_mode,
        "image_size": image_size,
        "context_size": context_size,
        "tile_stride": tile_stride,
        "tile_train_sampling": tile_train_sampling,
        "roi_threshold": roi_threshold,
        "min_roi_ratio": min_roi_ratio,
        "augmentation_profile": augmentation_profile,
    }
    if score_region is not None:
        observed["score_region"] = score_region
    if score_smoothing is not None:
        observed["score_smoothing"] = score_smoothing
    if score_image is not None:
        observed["score_image"] = score_image
    if topk_fraction is not None:
        observed["topk_fraction"] = topk_fraction

    mismatches = [
        f"{name}={value!r} expected {getattr(expected, name)!r}"
        for name, value in observed.items()
        if not _contract_value_matches(value, getattr(expected, name))
    ]
    if mismatches:
        raise ValueError(
            "Non-canonical Feature-AE preprocessing is not allowed for comparable "
            "bootstrap/lifecycle candidates. Use --allow-noncanonical-preprocessing "
            "only for tests or local dev. Mismatches: "
            + "; ".join(mismatches)
        )


def _contract_value_matches(value: object, expected: object) -> bool:
    if isinstance(expected, float):
        try:
            return abs(float(value) - expected) <= 1e-9
        except (TypeError, ValueError):
            return False
    return value == expected


__all__ = [
    "CANONICAL_FEATURE_AE_PREPROCESSING",
    "FEATURE_AE_BUSINESS_METRIC_PRIORITY",
    "FEATURE_AE_PREPROCESSING_CONTRACT_VERSION",
    "FeatureAEPreprocessingContract",
    "assert_canonical_feature_ae_preprocessing",
    "canonical_feature_ae_preprocessing_dict",
]
