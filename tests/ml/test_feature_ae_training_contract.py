from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from iqa.datasets import CALIBRATION_SET_ID, TiledFeatureAEDataset, is_calibration_sample, iter_manifest_image_samples, tile_boxes
from iqa.models.feature_ae import FEATURE_AE_MODEL_TYPE
from iqa.training.feature_ae import FeatureAETrainingConfig, train_feature_ae
from iqa.training.feature_ae_evaluation import (
    compute_binary_metrics,
    parse_layer_loss_weights,
    score_image_map,
    smooth_score_map,
    update_metric_best_checkpoints,
)


def test_tile_boxes_cover_image_with_overlap() -> None:
    assert tile_boxes(48, 48, tile_size=32, stride=16) == [
        (0, 0, 32, 32),
        (16, 0, 48, 32),
        (0, 16, 32, 48),
        (16, 16, 48, 48),
    ]


def test_tiled_dataset_keeps_roi_and_gt_masks_separate(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_path = image_root / "Casting_class1" / "test" / "defective" / "part.jpg"
    gt_path = image_root / "Casting_class1" / "test" / "defective" / "part_mask.png"
    roi_path = tmp_path / "roi.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (48, 48), "gray").save(image_path)
    _save_mask(gt_path, (48, 48), box=(20, 20, 28, 28))
    _save_mask(roi_path, (48, 48), box=(0, 0, 48, 48))
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {
                "image_ids": "img_defect",
                "relative_paths": "Casting_class1/test/defective/part.jpg",
                "split_set": "test",
                "label": "defective",
                "is_defective": "true",
                "mask_path": "Casting_class1/test/defective/part_mask.png",
            }
        ],
    )
    dataset = TiledFeatureAEDataset(
        manifest,
        image_root,
        tile_size=32,
        context_size=64,
        tile_stride=16,
        roi_masks={"img_defect": roi_path},
    )

    item = dataset[0]
    assert item["roi_mask"].sum() > item["gt_mask"].sum()
    assert item["gt_mask"].sum() > 0


def test_manifest_parser_accepts_singular_gt_mask_path_for_progressive_eval(tmp_path: Path) -> None:
    manifest = tmp_path / "evaluation_set.csv"
    _write_manifest(
        manifest,
        [
            {
                "image_id": "img_defect",
                "relative_path": "Casting_class1/test/defective/part.jpg",
                "split_set": "progressive_eval",
                "label": "defective",
                "is_defective": "true",
                "gt_mask_path": "Casting_class1/test/defective/part_mask.png",
            }
        ],
    )

    sample = iter_manifest_image_samples(manifest)[0]

    assert sample.gt_mask_path == "Casting_class1/test/defective/part_mask.png"


def test_tiled_dataset_resolves_progressive_eval_gt_mask_from_hss_iad_path(tmp_path: Path) -> None:
    image_root = tmp_path / "data" / "raw" / "hss-iad"
    image_path = image_root / "Casting_class1" / "test" / "defective" / "part.jpg"
    mask_path = image_root / "Casting_class1" / "test" / "defective" / "part_mask.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), (128, 128, 128)).save(image_path)
    _save_mask(mask_path, (32, 32), (8, 8, 16, 16))
    manifest = tmp_path / ".cache" / "iqa" / "replay_lifecycle" / "run" / "evaluation_set.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    _write_manifest(
        manifest,
        [
            {
                "image_id": "part",
                "relative_path": "Casting_class1/test/defective/part.jpg",
                "split_set": "progressive_eval",
                "label": "defective",
                "is_defective": "true",
                "gt_mask_path": "../raw/hss-iad/Casting_class1/test/defective/part_mask.png",
            }
        ],
    )

    item = TiledFeatureAEDataset(manifest, image_root, tile_size=32, context_size=64, tile_stride=32)[0]

    assert item["gt_mask"].sum() > 0


def test_train_normal_without_gt_uses_empty_defect_mask(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_path = image_root / "Casting_class1" / "train" / "good" / "part.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (32, 32), "gray").save(image_path)
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {
                "image_ids": "img_good",
                "relative_paths": "Casting_class1/train/good/part.jpg",
                "split_set": "train",
                "label": "good",
                "is_defective": "false",
            }
        ],
    )
    dataset = TiledFeatureAEDataset(manifest, image_root, tile_size=32, context_size=64, train_only_normal=True)

    assert dataset[0]["gt_mask"].sum() == 0


def test_calibration_set_is_recognized_and_excluded_from_training(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_path = image_root / "Casting_class1" / "calibration" / "good" / "part.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (32, 32), "gray").save(image_path)
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {
                "image_ids": "img_cal",
                "relative_paths": "Casting_class1/calibration/good/part.jpg",
                "split_set": CALIBRATION_SET_ID,
                "label": "good",
                "is_defective": "false",
            }
        ],
    )
    samples = iter_manifest_image_samples(manifest)

    assert is_calibration_sample(samples[0]) is True
    with pytest.raises(ValueError, match="No Feature-AE tiles"):
        TiledFeatureAEDataset(manifest, image_root, tile_size=32, context_size=64, train_only_normal=True)


def test_training_rejects_replay_candidate_without_versions(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Replay candidates require version metadata"):
        train_feature_ae(
            FeatureAETrainingConfig(
                manifest_path=tmp_path / "missing.csv",
                image_root=tmp_path,
                output_checkpoint=tmp_path / "checkpoint.pt",
                scenario_id="production_replay_natural",
            )
        )


def test_layer_weight_parser_and_score_helpers() -> None:
    assert parse_layer_loss_weights(["layer2=0.65", "layer3=0.35"]) == {"layer2": 0.65, "layer3": 0.35}
    score_map = np.zeros((5, 5), dtype=np.float32)
    score_map[2, 2] = 10.0
    assert smooth_score_map(score_map, "median3")[2, 2] == 0.0
    assert score_image_map(score_map, mode="topk_mean", topk_fraction=0.04) == 10.0


def test_metric_best_checkpoint_aliases(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint_epoch_001.pt"
    torch.save({"model_type": FEATURE_AE_MODEL_TYPE, "state_dict": {}}, checkpoint)

    best = update_metric_best_checkpoints(
        run_dir=tmp_path,
        candidate_checkpoint=checkpoint,
        metrics={"image_ap": 0.7, "image_auroc": 0.6, "pixel_ap": 0.4, "pixel_aupimo_1e-5_1e-3": 0.5},
        epoch=1,
    )

    assert best["image_ap"]["value"] == 0.7
    assert (tmp_path / "checkpoint_best_image.pt").exists()
    assert (tmp_path / "checkpoint_best_localization.pt").exists()
    assert (tmp_path / "checkpoint.pt").exists()


def test_checkpoint_pt_follows_business_metric_priority(tmp_path: Path) -> None:
    image_checkpoint = tmp_path / "checkpoint_epoch_001.pt"
    aupimo_checkpoint = tmp_path / "checkpoint_epoch_002.pt"
    image_checkpoint.write_bytes(b"image-ap")
    aupimo_checkpoint.write_bytes(b"aupimo")

    update_metric_best_checkpoints(
        run_dir=tmp_path,
        candidate_checkpoint=image_checkpoint,
        metrics={"image_ap": 0.9},
        epoch=1,
    )
    update_metric_best_checkpoints(
        run_dir=tmp_path,
        candidate_checkpoint=aupimo_checkpoint,
        metrics={"pixel_aupimo_1e-5_1e-3": 0.1},
        epoch=2,
    )

    assert (tmp_path / "checkpoint.pt").read_bytes() == b"aupimo"


def test_aupimo_uses_good_quantile_pimo_not_flat_low_fpr_roc() -> None:
    metrics = compute_binary_metrics(
        image_labels=[False, True],
        image_scores=[0.1, 0.9],
        pixel_labels=[
            np.zeros((2, 2), dtype=np.uint8),
            np.asarray([[1, 0], [0, 0]], dtype=np.uint8),
        ],
        pixel_scores=[
            np.zeros((2, 2), dtype=np.float32),
            np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        ],
    )

    assert metrics["pixel_aupimo_1e-5_1e-3"] == 1.0


def test_aupimo_ignores_defective_images_without_positive_gt_mask() -> None:
    metrics = compute_binary_metrics(
        image_labels=[False, True],
        image_scores=[0.1, 0.9],
        pixel_labels=[
            np.zeros((2, 2), dtype=np.uint8),
            np.zeros((2, 2), dtype=np.uint8),
        ],
        pixel_scores=[
            np.zeros((2, 2), dtype=np.float32),
            np.ones((2, 2), dtype=np.float32),
        ],
    )

    assert metrics["pixel_aupimo_1e-5_1e-3"] is None


def test_tiny_feature_ae_training_smoke(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    for index in range(2):
        image_path = image_root / "Casting_class1" / "train" / "good" / f"part_{index}.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), (128 + index, 128, 128)).save(image_path)
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {
                "image_ids": f"img_{index}",
                "relative_paths": f"Casting_class1/train/good/part_{index}.jpg",
                "split_set": "train",
                "label": "good",
                "is_defective": "false",
            }
            for index in range(2)
        ],
    )

    result = train_feature_ae(
        FeatureAETrainingConfig(
            manifest_path=manifest,
            image_root=image_root,
            output_checkpoint=tmp_path / "run" / "checkpoint.pt",
            image_size=32,
            context_size=64,
            tile_stride=32,
            batch_size=1,
            epochs=1,
            repeat_factor=1,
            val_fraction=0.0,
            max_steps=1,
            early_stopping_patience=0,
            allow_noncanonical_preprocessing=True,
        )
    )

    assert result["steps"] == 1
    assert (tmp_path / "run" / "checkpoint_last.pt").exists()
    assert (tmp_path / "run" / "params.json").exists()


def test_metric_early_stopping_uses_business_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image_root = tmp_path / "images"
    for index in range(2):
        image_path = image_root / "Casting_class1" / "train" / "good" / f"part_{index}.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), (128 + index, 128, 128)).save(image_path)
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {
                "image_ids": f"img_{index}",
                "relative_paths": f"Casting_class1/train/good/part_{index}.jpg",
                "split_set": "train",
                "label": "good",
                "is_defective": "false",
            }
            for index in range(2)
        ],
    )

    def fake_eval(config: object) -> dict[str, object]:
        return {"metrics": {"image_ap": 0.5}}

    monkeypatch.setattr("iqa.training.feature_ae.evaluate_feature_ae_checkpoint", fake_eval)

    result = train_feature_ae(
        FeatureAETrainingConfig(
            manifest_path=manifest,
            image_root=image_root,
            output_checkpoint=tmp_path / "run" / "checkpoint.pt",
            image_size=32,
            context_size=64,
            tile_stride=32,
            batch_size=1,
            epochs=8,
            repeat_factor=1,
            val_fraction=0.0,
            early_stopping_patience=0,
            metric_eval_manifest_path=manifest,
            metric_eval_every_epochs=1,
            metric_early_stopping_patience=2,
            allow_noncanonical_preprocessing=True,
        )
    )

    assert result["metric_early_stopped"] is True
    assert result["best_business_metric"] == "image_ap"
    assert result["steps"] == 6


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "image_ids",
        "image_id",
        "relative_paths",
        "relative_path",
        "split_set",
        "label",
        "is_defective",
        "mask_path",
        "gt_mask_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_mask(path: Path, size: tuple[int, int], box: tuple[int, int, int, int]) -> None:
    array = np.zeros(size[::-1], dtype=np.uint8)
    x0, y0, x1, y1 = box
    array[y0:y1, x0:x1] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)
