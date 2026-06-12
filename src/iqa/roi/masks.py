"""Adapters for fixed ROI segmenter mask outputs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


MASK_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class RoiMaskLookup:
    masks: dict[str, Path]
    status: dict[str, str]


def load_roi_mask_lookup(paths: list[str | Path] | tuple[str | Path, ...] | None) -> RoiMaskLookup:
    """Load ROI mask paths indexed by image id or relative path.

    A lookup source can be either a CSV file with `image_id`/`relative_path` and
    `roi_mask_path`/`mask_path`, or a directory containing mask images named with
    the same stem as the inspected image.
    """

    masks: dict[str, Path] = {}
    status: dict[str, str] = {}
    for source in paths or []:
        path = Path(source)
        if path.is_dir():
            for mask_path in path.rglob("*"):
                if mask_path.suffix.lower() in MASK_SUFFIXES:
                    masks.setdefault(mask_path.stem, mask_path)
            continue
        if path.suffix.lower() == ".csv":
            _load_roi_csv(path, masks, status)
            continue
        if path.exists():
            masks.setdefault(path.stem, path)
    return RoiMaskLookup(masks=masks, status=status)


def _load_roi_csv(path: Path, masks: dict[str, Path], status: dict[str, str]) -> None:
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            mask_value = row.get("roi_mask_path") or row.get("mask_path") or row.get("path") or ""
            if not mask_value:
                continue
            mask_path = Path(mask_value)
            if not mask_path.is_absolute():
                mask_path = path.parent / mask_path
            keys = [row.get("image_id") or "", row.get("relative_path") or "", Path(row.get("relative_path") or "").stem]
            roi_status = row.get("roi_quality_status") or row.get("roi_status") or row.get("status") or ""
            for key in (key for key in keys if key):
                masks[key] = mask_path
                if roi_status:
                    status[key] = roi_status


__all__ = ["RoiMaskLookup", "load_roi_mask_lookup"]
