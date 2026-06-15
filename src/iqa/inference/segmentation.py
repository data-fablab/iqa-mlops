"""ROI segmenter image prediction for the fixed IQA runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as tvf

from iqa.models.segmentation import (
    load_roi_segmenter,
    load_roi_segmenter_checkpoint,
    mask_logits_from_output,
    surface_probability_from_logits,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_SEGMENTATION_IMAGE_SIZE = 384


@dataclass(frozen=True)
class RoiSegmentationPrediction:
    model_type: str
    checkpoint_path: str
    image_path: str
    roi_ratio: float
    roi_quality_status: str
    threshold: float
    image_size: int
    context_size: int
    mask_mode: str

    def to_dict(self) -> dict[str, str | float | int]:
        return asdict(self)


def predict_roi_image(
    image_path: str | Path,
    checkpoint_path: str | Path,
    *,
    image_size: int | None = None,
    context_size: int | None = None,
    threshold: float = 0.5,
    min_roi_ratio: float = 0.01,
    max_roi_ratio: float = 0.98,
    surface_class: int = 1,
    mask_mode: str = "argmax",
    device: str | torch.device = "cpu",
    output_mask: str | Path | None = None,
) -> RoiSegmentationPrediction:
    image_path = Path(image_path)
    checkpoint_path = Path(checkpoint_path)
    torch_device = torch.device(device)
    checkpoint = load_roi_segmenter_checkpoint(checkpoint_path, map_location="cpu")
    resolved_image_size = int(image_size or checkpoint.get("input_size", DEFAULT_SEGMENTATION_IMAGE_SIZE))
    resolved_context_size = int(context_size or checkpoint.get("context_size", resolved_image_size))
    num_classes = int(checkpoint.get("num_classes", 1))
    model = load_roi_segmenter(checkpoint_path, map_location=torch_device).to(torch_device)
    image = _load_rgb_tensor(image_path, image_size=resolved_image_size).to(torch_device)
    global_image = _load_rgb_tensor(image_path, image_size=resolved_context_size).to(torch_device)
    crop_box_mask = _full_crop_box_mask(resolved_context_size).to(dtype=image.dtype, device=torch_device)

    with torch.no_grad():
        output = model(image, global_image=global_image, crop_box_mask=crop_box_mask)
        logits = mask_logits_from_output(output)
        mask = _surface_mask_from_logits(
            logits,
            surface_class=surface_class,
            threshold=threshold,
            mask_mode=mask_mode,
            num_classes=num_classes,
        )

    roi_ratio = float(mask.float().mean().item())
    if output_mask is not None:
        _save_mask(mask, Path(output_mask))
    return RoiSegmentationPrediction(
        model_type=str(getattr(model, "model_type", "functional_unet_resnet18_det1_context2b")),
        checkpoint_path=str(checkpoint_path),
        image_path=str(image_path),
        roi_ratio=roi_ratio,
        roi_quality_status=_roi_status(roi_ratio, min_roi_ratio=min_roi_ratio, max_roi_ratio=max_roi_ratio),
        threshold=float(threshold),
        image_size=resolved_image_size,
        context_size=resolved_context_size,
        mask_mode=mask_mode,
    )


def _load_rgb_tensor(path: Path, *, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = _resize_letterbox_pil(image, int(image_size))
    tensor = tvf.to_tensor(image)
    tensor = tvf.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return tensor.unsqueeze(0)


def _resize_letterbox_pil(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    scale = int(size) / max(width, height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (int(size), int(size)), (0, 0, 0))
    left = (int(size) - resized_width) // 2
    top = (int(size) - resized_height) // 2
    canvas.paste(resized, (left, top))
    return canvas


def _full_crop_box_mask(size: int) -> torch.Tensor:
    return torch.ones((1, 1, int(size), int(size)), dtype=torch.float32)


def _surface_mask_from_logits(
    logits: torch.Tensor,
    *,
    surface_class: int,
    threshold: float,
    mask_mode: str,
    num_classes: int,
) -> torch.Tensor:
    if logits.shape[1] == 1 or num_classes == 1:
        probability = surface_probability_from_logits(logits, surface_class=surface_class)
        return probability >= float(threshold)
    if mask_mode == "argmax":
        return torch.argmax(F.softmax(logits, dim=1), dim=1, keepdim=True) == int(surface_class)
    if mask_mode == "threshold":
        probability = surface_probability_from_logits(logits, surface_class=surface_class)
        return probability >= float(threshold)
    raise ValueError(f"Unsupported ROI mask_mode: {mask_mode!r}")


def _save_mask(mask: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = mask.squeeze().to(dtype=torch.uint8, device="cpu") * 255
    Image.fromarray(array.numpy(), mode="L").save(path)


def _roi_status(roi_ratio: float, *, min_roi_ratio: float, max_roi_ratio: float) -> str:
    if roi_ratio <= 0.0 or roi_ratio >= 1.0:
        return "fail"
    if roi_ratio < float(min_roi_ratio) or roi_ratio > float(max_roi_ratio):
        return "warning"
    return "ok"


__all__ = ["RoiSegmentationPrediction", "predict_roi_image"]
