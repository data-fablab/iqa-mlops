"""Defect coverage gate for promotion decisions."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from iqa.datasets import CastingImageSample


def compute_defect_coverage(
    samples: list[CastingImageSample] | list[dict[str, Any]],
) -> float:
    """Calculate the fraction of source_classes with at least one defective sample.

    Args:
        samples: List of samples with source_class and is_defective fields.

    Returns:
        Defect coverage as a fraction [0.0, 1.0].
        Returns 0.0 if no samples or no source_classes exist.
    """
    if not samples:
        return 0.0

    classes_with_defects = set()
    all_classes = set()

    for sample in samples:
        if isinstance(sample, dict):
            source_class = sample.get("source_class", "")
            is_defective = sample.get("is_defective", False)
        else:
            source_class = sample.source_class
            is_defective = sample.is_defective

        if source_class:
            all_classes.add(source_class)
            if is_defective:
                classes_with_defects.add(source_class)

    if not all_classes:
        return 0.0

    return len(classes_with_defects) / len(all_classes)


def check_defect_coverage_gate(
    coverage: float,
    min_coverage: float = 0.95,
) -> dict[str, bool | float]:
    """Check if defect_coverage meets gate threshold.

    Args:
        coverage: Computed defect coverage value.
        min_coverage: Minimum acceptable coverage threshold.

    Returns:
        Dict with keys:
        - passed: bool, True if coverage >= min_coverage
        - coverage: float, the computed coverage value
        - threshold: float, the minimum threshold
    """
    return {
        "passed": coverage >= min_coverage,
        "coverage": coverage,
        "threshold": min_coverage,
    }


def compute_defect_coverage_from_manifest(
    manifest_path: Path | str,
    image_id_field: str = "image_id",
) -> float:
    """Compute defect_coverage from a CSV manifest file.

    The manifest must have columns: source_class and is_defective.

    Args:
        manifest_path: Path to manifest CSV.
        image_id_field: Field name to use as image identifier.

    Returns:
        Defect coverage as a fraction [0.0, 1.0].
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return 0.0

    samples: list[CastingImageSample] = []
    with manifest_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return 0.0

        for row in reader:
            image_id = row.get(image_id_field, "")
            source_class = row.get("source_class", "")
            is_defective_str = row.get("is_defective", "").lower()
            is_defective = is_defective_str in {"1", "true", "yes", "defective", "anomaly"}

            sample = CastingImageSample(
                image_id=image_id,
                relative_path=row.get("relative_path", ""),
                source_class=source_class,
                is_defective=is_defective,
            )
            samples.append(sample)

    return compute_defect_coverage(samples)


__all__ = [
    "compute_defect_coverage",
    "check_defect_coverage_gate",
    "compute_defect_coverage_from_manifest",
]
