"""Contract tests for model wrappers with homogeneous interface."""
from __future__ import annotations

import torch

from iqa.models.segmentation import FunctionalSurfaceUNetResNet18Det1Context2B
from iqa.models.feature_ae import ReverseDistillationGatedDualContextResNet18
from iqa.models.feature_ae.teacher import ResNetTeacherFeatures
from iqa.models.wrappers import ROISegmenterWrapper, TeacherWrapper, FeatureAEWrapper


def test_wrappers_are_importable() -> None:
    from iqa.models import wrappers

    assert hasattr(wrappers, "ROISegmenterWrapper")
    assert hasattr(wrappers, "TeacherWrapper")
    assert hasattr(wrappers, "FeatureAEWrapper")


def test_roi_segmenter_wrapper_exposes_load_and_predict() -> None:
    model = FunctionalSurfaceUNetResNet18Det1Context2B(pretrained=False)
    wrapper = ROISegmenterWrapper.load(model)

    assert hasattr(wrapper, "predict")
    assert callable(wrapper.predict)

    images = torch.randn(1, 3, 256, 256)
    result = wrapper.predict(images)

    assert isinstance(result, dict)
    assert "mask_logits" in result
    assert result["mask_logits"].shape == (1, 1, 256, 256)


def test_teacher_wrapper_exposes_load_and_predict() -> None:
    model = ResNetTeacherFeatures(pretrained=False)
    wrapper = TeacherWrapper.load(model)

    assert hasattr(wrapper, "predict")
    assert callable(wrapper.predict)

    images = torch.randn(1, 3, 256, 256)
    result = wrapper.predict(images)

    assert isinstance(result, dict)
    assert "layer2" in result or "layer3" in result


def test_feature_ae_wrapper_exposes_load_and_predict() -> None:
    model = ReverseDistillationGatedDualContextResNet18()
    wrapper = FeatureAEWrapper.load(model)

    assert hasattr(wrapper, "predict")
    assert callable(wrapper.predict)

    images = torch.randn(1, 3, 256, 256)
    result = wrapper.predict(images)

    assert isinstance(result, dict)
    assert "layer2" in result or "layer3" in result
