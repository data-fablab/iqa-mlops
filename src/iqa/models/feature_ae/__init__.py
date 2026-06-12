"""RD Feature-AE runtime package."""

from iqa.models.feature_ae.losses import feature_anomaly_map, feature_reconstruction_loss
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
    "ReverseDistillationGatedDualContextResNet18",
    "ResNetTeacherFeatures",
    "feature_anomaly_map",
    "feature_reconstruction_loss",
    "load_rd_feature_ae_gated",
    "normalize_feature_layers",
]
