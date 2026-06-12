from __future__ import annotations

import pytest
import torch

from iqa.models.segmentation import (
    ROI_SEGMENTER_MODEL_TYPE,
    FunctionalSurfaceUNetResNet18Det1Context2B,
    build_segmentation_model,
    mask_logits_from_output,
)


def test_build_segmentation_model_retained_type() -> None:
    model = build_segmentation_model(ROI_SEGMENTER_MODEL_TYPE)

    assert isinstance(model, FunctionalSurfaceUNetResNet18Det1Context2B)


def test_build_segmentation_model_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="Unsupported segmentation model_type"):
        build_segmentation_model("old_experimental_unet")


def test_segmentation_forward_with_context_inputs() -> None:
    model = build_segmentation_model(pretrained=False)
    model.eval()
    image = torch.randn(1, 3, 64, 64)
    global_image = torch.randn(1, 3, 64, 64)
    crop_box_mask = torch.ones(1, 1, 64, 64)

    with torch.no_grad():
        output = model(image, global_image=global_image, crop_box_mask=crop_box_mask)

    assert mask_logits_from_output(output).shape == (1, 1, 64, 64)
    assert output["objectness_logits"].shape == (1, 1)
    assert output["bbox"].shape == (1, 4)
