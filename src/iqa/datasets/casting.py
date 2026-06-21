"""Casting datasets built from IQA metadata manifests."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FEATURE_AE_TILE_SIZE = 384
FEATURE_AE_CONTEXT_SIZE = 768
FEATURE_AE_PREPROCESSING_MODES = ("letterbox", "tiled_context")
CALIBRATION_SET_ID = "calibration_set_v001"
VALIDATION_SET_ID = "validation_set_v001"
FEATURE_AE_EXCLUDED_TRAIN_SETS = (CALIBRATION_SET_ID, VALIDATION_SET_ID, "incident")


@dataclass(frozen=True)
class CastingImageSample:
    image_id: str
    relative_path: str
    event_id: str = ""
    source_class: str = ""
    split_set: str = ""
    label: str = "good"
    is_defective: bool = False
    scenario_id: str = ""
    dataset_version: str = ""
    gt_mask_path: str = ""
    oracle_verdict: str = ""
    train_eligible: bool = False
    train_eligibility_source: str = ""
    quarantine_reason: str = ""


@dataclass(frozen=True)
class TileRecord:
    sample: CastingImageSample
    tile_box: tuple[int, int, int, int]
    context_box: tuple[int, int, int, int]
    image_size: tuple[int, int]


class ResizeLetterbox:
    """Resize while preserving aspect ratio, then pad to a square canvas."""

    def __init__(self, size: int, fill: int | tuple[int, int, int] = 0) -> None:
        self.size = int(size)
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        width, height = image.size
        scale = self.size / max(width, height)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        resized = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        canvas.paste(resized, ((self.size - new_width) // 2, (self.size - new_height) // 2))
        return canvas


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "defective", "anomaly"}


def _split_pipe(value: str | None) -> list[str]:
    return [part for part in (value or "").split("|") if part]


def _normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype)[:, None, None]
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype)[:, None, None]
    return (tensor - mean) / std


def image_to_tensor(image: Image.Image, *, image_size: int = FEATURE_AE_TILE_SIZE) -> torch.Tensor:
    image = ResizeLetterbox(image_size)(image)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return _normalize_tensor(torch.from_numpy(array).permute(2, 0, 1))


def load_image_tensor(path: str | Path, *, image_size: int = FEATURE_AE_TILE_SIZE) -> torch.Tensor:
    return image_to_tensor(Image.open(path).convert("RGB"), image_size=image_size)


def image_crop_to_tensor(
    image: Image.Image,
    box: tuple[int, int, int, int],
    *,
    output_size: int,
    fill: int | tuple[int, int, int] = 0,
) -> torch.Tensor:
    x0, y0, x1, y1 = box
    width, height = image.size
    crop = Image.new("RGB", (x1 - x0, y1 - y0), fill)
    source_box = (max(0, x0), max(0, y0), min(width, x1), min(height, y1))
    if source_box[2] > source_box[0] and source_box[3] > source_box[1]:
        patch = image.crop(source_box)
        crop.paste(patch, (source_box[0] - x0, source_box[1] - y0))
    if crop.size != (output_size, output_size):
        crop = crop.resize((output_size, output_size), Image.Resampling.BILINEAR)
    array = np.asarray(crop, dtype=np.float32) / 255.0
    return _normalize_tensor(torch.from_numpy(array).permute(2, 0, 1))


def load_mask_tensor(path: str | Path, *, size: tuple[int, int] | None = None, threshold: float = 0.5) -> torch.Tensor:
    mask = Image.open(path).convert("L")
    if size is not None and mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    array = np.asarray(mask, dtype=np.float32) / 255.0
    return (torch.from_numpy(array) >= float(threshold)).float().unsqueeze(0)


def _resolve_indexed(parts: list[str], index: int) -> str:
    return parts[index] if index < len(parts) else ""


def iter_manifest_image_samples(manifest_path: str | Path) -> list[CastingImageSample]:
    samples: list[CastingImageSample] = []
    with Path(manifest_path).open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            image_ids = _split_pipe(row.get("image_ids") or row.get("image_id"))
            relative_paths = _split_pipe(row.get("relative_paths") or row.get("relative_path"))
            gt_paths = _split_pipe(
                row.get("gt_mask_paths")
                or row.get("gt_mask_path")
                or row.get("mask_paths")
                or row.get("mask_path")
            )
            for index, relative_path in enumerate(relative_paths):
                image_id = _resolve_indexed(image_ids, index) or Path(relative_path).stem
                samples.append(
                    CastingImageSample(
                        image_id=image_id,
                        relative_path=relative_path,
                        event_id=row.get("event_id") or row.get("piece_event_id") or "",
                        source_class=row.get("source_class") or row.get("category") or "",
                        split_set=row.get("split_set") or row.get("split") or "",
                        label=row.get("label") or "good",
                        is_defective=_truthy(row.get("is_defective") or row.get("defective") or ""),
                        scenario_id=row.get("scenario_id") or "",
                        dataset_version=row.get("dataset_version") or row.get("bootstrap_dataset_version") or "",
                        gt_mask_path=_resolve_indexed(gt_paths, index),
                        oracle_verdict=row.get("oracle_verdict") or "",
                        train_eligible=_truthy(row.get("train_eligible") or ""),
                        train_eligibility_source=row.get("train_eligibility_source") or "",
                        quarantine_reason=row.get("quarantine_reason") or "",
                    )
                )
    return samples


def tile_boxes(width: int, height: int, *, tile_size: int, stride: int) -> list[tuple[int, int, int, int]]:
    if tile_size <= 0 or stride <= 0:
        raise ValueError("tile_size and stride must be positive.")

    def starts(length: int) -> list[int]:
        if length <= tile_size:
            return [0]
        values = list(range(0, max(1, length - tile_size + 1), stride))
        last = length - tile_size
        if values[-1] != last:
            values.append(last)
        return values

    return [(x, y, x + tile_size, y + tile_size) for y in starts(height) for x in starts(width)]


def centered_box(box: tuple[int, int, int, int], *, size: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    half = size // 2
    return (cx - half, cy - half, cx - half + size, cy - half + size)


def _crop_mask(mask: torch.Tensor, box: tuple[int, int, int, int], output_size: int) -> torch.Tensor:
    x0, y0, x1, y1 = box
    canvas = torch.zeros((1, y1 - y0, x1 - x0), dtype=mask.dtype)
    _, height, width = mask.shape
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
    if sx1 > sx0 and sy1 > sy0:
        canvas[:, sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = mask[:, sy0:sy1, sx0:sx1]
    if canvas.shape[-1] != output_size or canvas.shape[-2] != output_size:
        canvas = F.interpolate(canvas.unsqueeze(0), size=(output_size, output_size), mode="nearest").squeeze(0)
    return canvas


class CastingImageDataset(Dataset):
    """Image dataset backed by Casting metadata CSV files."""

    def __init__(
        self,
        manifest_path: str | Path,
        image_root: str | Path,
        *,
        image_size: int = FEATURE_AE_TILE_SIZE,
        context_size: int = FEATURE_AE_CONTEXT_SIZE,
        preprocessing_mode: str = "letterbox",
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.image_root = Path(image_root)
        self.image_size = int(image_size)
        self.context_size = int(context_size)
        if preprocessing_mode not in FEATURE_AE_PREPROCESSING_MODES:
            raise ValueError(f"Unsupported preprocessing_mode {preprocessing_mode!r}.")
        self.preprocessing_mode = preprocessing_mode
        self.samples = iter_manifest_image_samples(self.manifest_path)
        if not self.samples:
            raise ValueError(f"No image paths found in manifest: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | bool]:
        sample = self.samples[index]
        image_path = self.image_root / sample.relative_path
        if not image_path.exists():
            raise FileNotFoundError(f"Casting image not found: {image_path}")
        image = Image.open(image_path).convert("RGB")
        context_size = self.context_size if self.preprocessing_mode == "tiled_context" else self.image_size
        return {
            "image": image_to_tensor(image, image_size=self.image_size),
            "context_image": image_to_tensor(image, image_size=context_size),
            "image_id": sample.image_id,
            "relative_path": sample.relative_path,
            "label": sample.label,
            "is_defective": sample.is_defective,
        }


class TiledFeatureAEDataset(Dataset):
    """Overlapping 384/768 Feature-AE tiles from IQA image manifests."""

    def __init__(
        self,
        manifest_path: str | Path,
        image_root: str | Path,
        *,
        tile_size: int = FEATURE_AE_TILE_SIZE,
        context_size: int = FEATURE_AE_CONTEXT_SIZE,
        tile_stride: int = FEATURE_AE_TILE_SIZE // 2,
        repeat_factor: int = 1,
        roi_masks: dict[str, Path] | None = None,
        roi_status: dict[str, str] | None = None,
        gt_masks: dict[str, Path] | None = None,
        roi_threshold: float = 0.5,
        min_roi_ratio: float = 0.0,
        train_only_normal: bool = False,
        reject_roi_not_ok: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.image_root = Path(image_root)
        self.tile_size = int(tile_size)
        self.context_size = int(context_size)
        self.tile_stride = int(tile_stride)
        self.roi_masks = roi_masks or {}
        self.roi_status = roi_status or {}
        self.gt_masks = gt_masks or {}
        self.roi_threshold = float(roi_threshold)

        records: list[TileRecord] = []
        samples = iter_manifest_image_samples(self.manifest_path)
        if train_only_normal:
            samples = [sample for sample in samples if _is_train_normal(sample, reject_validation=True)]
        for sample in samples:
            status = self.roi_status.get(sample.image_id) or self.roi_status.get(sample.relative_path)
            if reject_roi_not_ok and status and status.lower() != "ok":
                continue
            image_path = self.image_root / sample.relative_path
            if not image_path.exists():
                raise FileNotFoundError(f"Casting image not found: {image_path}")
            with Image.open(image_path) as image:
                width, height = image.size
            roi_mask = self._load_roi_mask(sample, (width, height))
            for box in tile_boxes(width, height, tile_size=self.tile_size, stride=self.tile_stride):
                if min_roi_ratio and roi_mask is not None:
                    roi_ratio = float(_crop_mask(roi_mask, box, self.tile_size).mean())
                    if roi_ratio < float(min_roi_ratio):
                        continue
                records.append(TileRecord(sample, box, centered_box(box, size=self.context_size), (width, height)))
        if repeat_factor > 1:
            records = records * int(repeat_factor)
        if not records:
            raise ValueError(f"No Feature-AE tiles found in manifest: {self.manifest_path}")
        self.records = records

    def _load_roi_mask(self, sample: CastingImageSample, size: tuple[int, int]) -> torch.Tensor | None:
        path = self.roi_masks.get(sample.image_id) or self.roi_masks.get(sample.relative_path)
        if path is None:
            return None
        return load_mask_tensor(path, size=size, threshold=self.roi_threshold)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | bool | tuple[int, int, int, int]]:
        record = self.records[index]
        sample = record.sample
        image = Image.open(self.image_root / sample.relative_path).convert("RGB")
        roi_mask = self._load_roi_mask(sample, record.image_size)
        gt_path = self.gt_masks.get(sample.image_id) or self.gt_masks.get(sample.relative_path)
        if gt_path is None and sample.gt_mask_path:
            gt_path = self.image_root / sample.gt_mask_path
        gt_mask = (
            load_mask_tensor(gt_path, size=record.image_size)
            if gt_path
            else torch.zeros((1, record.image_size[1], record.image_size[0]))
        )
        tile_roi = (
            _crop_mask(roi_mask, record.tile_box, self.tile_size)
            if roi_mask is not None
            else torch.ones((1, self.tile_size, self.tile_size))
        )
        return {
            "image": image_crop_to_tensor(image, record.tile_box, output_size=self.tile_size),
            "context_image": image_crop_to_tensor(image, record.context_box, output_size=self.context_size),
            "roi_mask": tile_roi,
            "gt_mask": _crop_mask(gt_mask, record.tile_box, self.tile_size),
            "image_id": sample.image_id,
            "relative_path": sample.relative_path,
            "event_id": sample.event_id,
            "source_class": sample.source_class,
            "scenario_id": sample.scenario_id,
            "label": sample.label,
            "is_defective": sample.is_defective,
            "tile_box": record.tile_box,
            "image_size": record.image_size,
        }


def _is_train_normal(sample: CastingImageSample, *, reject_validation: bool) -> bool:
    split = sample.split_set.lower()
    label = sample.label.lower()
    if reject_validation and _is_excluded_feature_ae_train_split(split):
        return False
    return not sample.is_defective and label in {"good", "normal", "conforme"}


def _is_excluded_feature_ae_train_split(split: str) -> bool:
    return any(excluded in split for excluded in FEATURE_AE_EXCLUDED_TRAIN_SETS)


def is_calibration_sample(sample: CastingImageSample) -> bool:
    split = sample.split_set.lower()
    return CALIBRATION_SET_ID in split and not sample.is_defective and sample.label.lower() in {"good", "normal", "conforme"}


def validate_good_only_samples(samples: Iterable[CastingImageSample]) -> None:
    invalid = [sample.image_id for sample in samples if not _is_train_normal(sample, reject_validation=True)]
    if invalid:
        raise ValueError(f"Feature-AE training accepts only normal non-validation samples: {invalid[:5]}")


__all__ = [
    "FEATURE_AE_CONTEXT_SIZE",
    "CALIBRATION_SET_ID",
    "FEATURE_AE_EXCLUDED_TRAIN_SETS",
    "FEATURE_AE_PREPROCESSING_MODES",
    "FEATURE_AE_TILE_SIZE",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "VALIDATION_SET_ID",
    "CastingImageDataset",
    "CastingImageSample",
    "ResizeLetterbox",
    "TileRecord",
    "TiledFeatureAEDataset",
    "centered_box",
    "image_crop_to_tensor",
    "image_to_tensor",
    "is_calibration_sample",
    "iter_manifest_image_samples",
    "load_image_tensor",
    "load_mask_tensor",
    "tile_boxes",
    "validate_good_only_samples",
]
