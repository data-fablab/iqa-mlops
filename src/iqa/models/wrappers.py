"""Homogeneous inference wrappers for ML models."""
from __future__ import annotations

from typing import TypeVar

import torch
from torch import nn

from iqa.models.segmentation import FunctionalSurfaceUNetResNet18Det1Context2B
from iqa.models.feature_ae import ReverseDistillationGatedDualContextResNet18
from iqa.models.feature_ae.teacher import ResNetTeacherFeatures

T = TypeVar("T", bound=nn.Module)


class ROISegmenterWrapper:
    """Wrapper for ROI segmenter with homogeneous inference interface."""

    def __init__(self, model: FunctionalSurfaceUNetResNet18Det1Context2B) -> None:
        self.model = model
        self.model.eval()

    @classmethod
    def load(cls, model: FunctionalSurfaceUNetResNet18Det1Context2B) -> ROISegmenterWrapper:
        """Load a ROI segmenter model into the wrapper."""
        return cls(model)

    def predict(
        self,
        images: torch.Tensor,
        *,
        global_image: torch.Tensor | None = None,
        crop_box_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict ROI segmentation."""
        with torch.no_grad():
            return self.model(images, global_image=global_image, crop_box_mask=crop_box_mask)


class TeacherWrapper:
    """Wrapper for Teacher ResNet18 with homogeneous inference interface."""

    def __init__(self, model: ResNetTeacherFeatures) -> None:
        self.model = model
        self.model.eval()

    @classmethod
    def load(cls, model: ResNetTeacherFeatures) -> TeacherWrapper:
        """Load a Teacher model into the wrapper."""
        return cls(model)

    def predict(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract features from Teacher ResNet18."""
        with torch.no_grad():
            return self.model(images)


class FeatureAEWrapper:
    """Wrapper for Feature-AE with homogeneous inference interface."""

    def __init__(self, model: ReverseDistillationGatedDualContextResNet18) -> None:
        self.model = model
        self.model.eval()

    @classmethod
    def load(cls, model: ReverseDistillationGatedDualContextResNet18) -> FeatureAEWrapper:
        """Load a Feature-AE model into the wrapper."""
        return cls(model)

    def predict(
        self,
        images: torch.Tensor,
        context_images: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict with Feature-AE."""
        with torch.no_grad():
            return self.model(images, context_images=context_images)


__all__ = ["ROISegmenterWrapper", "TeacherWrapper", "FeatureAEWrapper"]
