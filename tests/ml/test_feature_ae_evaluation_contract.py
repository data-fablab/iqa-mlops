from __future__ import annotations

from pathlib import Path

import pytest

from iqa.training.feature_ae_evaluation import _load_gt_mask_lookup


def test_gt_mask_lookup_prefers_image_root_for_relative_masks(tmp_path: Path) -> None:
    image_root = tmp_path / "source_datasets" / "hss-iad"
    mask = image_root / "Casting_class1" / "ground_truth" / "defective" / "part_mask.png"
    mask.parent.mkdir(parents=True)
    mask.write_bytes(b"runtime-mask")
    manifest_dir = tmp_path / "data" / "validation"
    fallback_mask = manifest_dir / "Casting_class1" / "ground_truth" / "defective" / "part_mask.png"
    fallback_mask.parent.mkdir(parents=True)
    fallback_mask.write_bytes(b"fallback-mask")
    manifest = manifest_dir / "gt_masks.csv"
    manifest.write_text(
        "image_id,relative_path,gt_mask_path\n"
        "img_001,Casting_class1/test/defective/part.jpg,Casting_class1/ground_truth/defective/part_mask.png\n",
        encoding="utf-8",
    )

    lookup = _load_gt_mask_lookup(manifest, image_root=image_root)

    assert lookup["img_001"] == mask
    assert lookup["Casting_class1/test/defective/part.jpg"] == mask


def test_gt_mask_lookup_fails_with_actionable_missing_mask_message(tmp_path: Path) -> None:
    image_root = tmp_path / "source_datasets" / "hss-iad"
    manifest = tmp_path / "gt_masks.csv"
    manifest.write_text(
        "image_id,gt_mask_path\n"
        "img_001,Casting_class1/ground_truth/defective/missing_mask.png\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="missing_mask.png"):
        _load_gt_mask_lookup(manifest, image_root=image_root)
