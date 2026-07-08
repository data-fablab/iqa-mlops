"""Bootstrap ROI mask generation from IQA piece-event manifests."""

from __future__ import annotations

import csv
from pathlib import Path

from iqa.inference.segmentation import predict_roi_image
from iqa.roi.artifacts import RoiPredictionArtifact, RoiQualityStatus


BOOTSTRAP_SOURCE = "historical_bootstrap"


def generate_bootstrap_roi_predictions(
    *,
    manifest_path: str | Path,
    image_root: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    roi_model_version: str,
    dataset_version: str = "bootstrap_v001",
    scenario_id: str = "bootstrap_v001",
    device: str = "cpu",
    limit: int | None = None,
) -> list[RoiPredictionArtifact]:
    manifest_path = Path(manifest_path)
    image_root = Path(image_root)
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    _validate_output_dir(output_dir)
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[RoiPredictionArtifact] = []
    with manifest_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows[:limit]:
        _validate_bootstrap_row(row)
        event_id = _required(row, "event_id")
        relative_paths = _split_manifest_field(_required(row, "relative_paths"))
        image_ids = _split_manifest_field(row.get("image_ids") or "")
        if not image_ids:
            image_ids = [Path(path).stem for path in relative_paths]
        if len(image_ids) != len(relative_paths):
            raise ValueError(f"Manifest row {event_id!r} has mismatched image_ids and relative_paths.")
        row_dataset_version = row.get("bootstrap_dataset_version") or dataset_version
        for image_id, relative_path in zip(image_ids, relative_paths, strict=True):
            image_path = image_root / relative_path
            mask_path = masks_dir / f"{event_id}_{image_id}_roi.png"
            prediction = predict_roi_image(image_path, checkpoint_path, device=device, output_mask=mask_path)
            artifacts.append(
                RoiPredictionArtifact(
                    piece_event_id=event_id,
                    image_id=image_id,
                    image_uri=relative_path.replace("\\", "/"),
                    roi_mask_uri=mask_path.as_posix(),
                    roi_model_version=roi_model_version,
                    roi_ratio=prediction.roi_ratio,
                    roi_quality_status=_roi_status(prediction.roi_quality_status),
                    source=BOOTSTRAP_SOURCE,
                    scenario_id=scenario_id,
                    dataset_version=row_dataset_version,
                )
            )
    _write_roi_predictions_index(output_dir / "roi_predictions.csv", artifacts)
    return artifacts


def _validate_output_dir(output_dir: Path) -> None:
    if "reports" in output_dir.parts:
        raise ValueError("Bootstrap ROI masks must be written under data/processed/roi, not reports/.")


def _validate_bootstrap_row(row: dict[str, str]) -> None:
    event_id = _required(row, "event_id")
    label = (row.get("label") or "").lower()
    is_defective = (row.get("is_defective") or "").lower()
    if label != "good" or is_defective == "true":
        raise ValueError(f"Bootstrap row {event_id!r} is not good-only.")
    bootstrap_role = row.get("bootstrap_role") or ""
    if bootstrap_role and not bootstrap_role.startswith("train_normal"):
        raise ValueError(f"Bootstrap row {event_id!r} has unsupported bootstrap_role.")


def _required(row: dict[str, str], column: str) -> str:
    value = row.get(column) or ""
    if not value:
        raise ValueError(f"Missing required manifest column value: {column}")
    return value


def _split_manifest_field(value: str) -> list[str]:
    return [item.strip() for item in str(value).split("|") if item.strip()]


def _roi_status(value: str) -> RoiQualityStatus:
    if value not in {"ok", "warning", "fail"}:
        raise ValueError(f"Unsupported ROI quality status: {value!r}")
    return value


def _write_roi_predictions_index(path: Path, artifacts: list[RoiPredictionArtifact]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(RoiPredictionArtifact.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for artifact in artifacts:
            writer.writerow(artifact.to_dict())


__all__ = ["BOOTSTRAP_SOURCE", "generate_bootstrap_roi_predictions"]
