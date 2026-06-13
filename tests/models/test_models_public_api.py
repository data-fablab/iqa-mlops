from __future__ import annotations

import iqa.models as iqa_models
import iqa.models.feature_ae as feature_ae
import iqa.models.segmentation as segmentation
from iqa.models.common import architectures as common_architectures


def test_models_root_exports_retained_runtime_domains() -> None:
    assert iqa_models.__all__ == ["feature_ae", "segmentation"]


def test_common_architectures_exports_only_shared_runtime_blocks() -> None:
    assert common_architectures.__all__ == ["build_resnet", "conv_block"]


def test_feature_ae_public_api_has_no_legacy_aliases() -> None:
    assert "FEATURE_AE_MODEL_TYPE" in feature_ae.__all__
    assert "DEFAULT_FEATURE_LAYERS" in feature_ae.__all__
    assert "feature_anomaly_map" in feature_ae.__all__
    assert not hasattr(feature_ae, "SUPPORTED_MODEL_TYPE")
    assert not hasattr(feature_ae, "DEFAULT_LAYERS")
    assert not hasattr(feature_ae, "feature_error_map")


def test_segmentation_public_api_is_minimal() -> None:
    assert segmentation.__all__ == [
        "FunctionalSurfaceUNetResNet18Det1Context2B",
        "ROI_SEGMENTER_MODEL_TYPE",
        "build_segmentation_model",
        "load_roi_segmenter_checkpoint",
        "load_roi_segmenter",
        "mask_logits_from_output",
        "replace_segmentation_head",
        "surface_probability_from_logits",
    ]
