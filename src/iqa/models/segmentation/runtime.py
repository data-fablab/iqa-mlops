"""ROI segmenter runtime helpers."""

from __future__ import annotations

import torch


def mask_logits_from_output(output: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(output, dict):
        return output["mask_logits"]
    return output


def replace_segmentation_head(model: torch.nn.Module, num_classes: int) -> None:
    if not hasattr(model, "out"):
        raise ValueError("Model does not expose an 'out' segmentation head.")
    old_head = model.out
    if not isinstance(old_head, torch.nn.Conv2d):
        raise ValueError(f"Unsupported segmentation head type: {type(old_head)}")
    model.out = torch.nn.Conv2d(old_head.in_channels, int(num_classes), kernel_size=old_head.kernel_size)


__all__ = ["mask_logits_from_output", "replace_segmentation_head"]
