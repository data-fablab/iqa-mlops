"""RD Feature-AE runtime package."""

from iqa.models.feature_ae.losses import feature_anomaly_map, feature_reconstruction_loss
from iqa.models.feature_ae.champion import (
    CHAMPION_FEATURE_AE_CONTRACT,
    FeatureAEChampionContract,
    apply_champion_roi,
    feature_layer_anomaly_maps,
    fuse_layer_anomaly_maps,
    fuse_numpy_layer_maps,
    load_roi_probability_map,
    reconstruct_tiled_feature_maps,
    score_numpy_map_topk,
    smooth_numpy_score_map,
)
from iqa.models.feature_ae.models import (
    DEFAULT_FEATURE_LAYERS,
    FEATURE_AE_MODEL_TYPE,
    SUPPORTED_TEACHER_BACKBONE,
    TEACHER_LAYER_CHANNELS,
    ReverseDistillationGatedDualContextResNet18,
    normalize_feature_layers,
)
from iqa.models.feature_ae.runtime import load_rd_feature_ae_gated
from iqa.models.feature_ae.teacher import ResNetTeacherFeatures

__all__ = [
    "DEFAULT_FEATURE_LAYERS",
    "FEATURE_AE_MODEL_TYPE",
    "SUPPORTED_TEACHER_BACKBONE",
    "TEACHER_LAYER_CHANNELS",
    "CHAMPION_FEATURE_AE_CONTRACT",
    "FeatureAEChampionContract",
    "ReverseDistillationGatedDualContextResNet18",
    "ResNetTeacherFeatures",
    "apply_champion_roi",
    "feature_anomaly_map",
    "feature_layer_anomaly_maps",
    "feature_reconstruction_loss",
    "fuse_layer_anomaly_maps",
    "fuse_numpy_layer_maps",
    "load_roi_probability_map",
    "load_rd_feature_ae_gated",
    "normalize_feature_layers",
    "reconstruct_tiled_feature_maps",
    "score_numpy_map_topk",
    "smooth_numpy_score_map",
]
