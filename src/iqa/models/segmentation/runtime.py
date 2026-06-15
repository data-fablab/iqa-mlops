"""ROI segmenter runtime helpers."""

from __future__ import annotations

from pathlib import Path

import torch

from iqa.models.segmentation.models import ROI_SEGMENTER_MODEL_TYPE, build_segmentation_model


def load_roi_segmenter_checkpoint(checkpoint_path: str | Path, *, map_location: str | torch.device = "cpu") -> dict:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"ROI segmenter checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        return {"state_dict": checkpoint}
    return checkpoint


def mask_logits_from_output(output: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(output, dict):
        return output["mask_logits"]
    return output


def load_roi_segmenter(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    pretrained: bool = False,
) -> torch.nn.Module:
    checkpoint = load_roi_segmenter_checkpoint(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict):
        model_type = checkpoint.get("model_type", ROI_SEGMENTER_MODEL_TYPE)
        if model_type != ROI_SEGMENTER_MODEL_TYPE:
            raise ValueError(f"Unsupported ROI segmenter model_type: {model_type!r}")
        num_classes = int(checkpoint.get("num_classes", 1))
        state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict") or checkpoint
    else:
        num_classes = 1
        state_dict = checkpoint
    model = build_segmentation_model(pretrained=pretrained)
    if num_classes != 1:
        replace_segmentation_head(model, num_classes)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def replace_segmentation_head(model: torch.nn.Module, num_classes: int) -> None:
    if not hasattr(model, "out"):
        raise ValueError("Model does not expose an 'out' segmentation head.")
    old_head = model.out
    if not isinstance(old_head, torch.nn.Conv2d):
        raise ValueError(f"Unsupported segmentation head type: {type(old_head)}")
    model.out = torch.nn.Conv2d(old_head.in_channels, int(num_classes), kernel_size=old_head.kernel_size)


def surface_probability_from_logits(logits: torch.Tensor, *, surface_class: int = 1) -> torch.Tensor:
    if logits.shape[1] == 1:
        return torch.sigmoid(logits[:, :1])
    return torch.softmax(logits, dim=1)[:, int(surface_class) : int(surface_class) + 1]


__all__ = [
    "load_roi_segmenter_checkpoint",
    "load_roi_segmenter",
    "mask_logits_from_output",
    "replace_segmentation_head",
    "surface_probability_from_logits",
]
