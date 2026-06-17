"""Build candidate datasets with AI safety filtering rules."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from iqa.datasets.casting import VALIDATION_SET_ID, CastingImageSample


@dataclass(frozen=True)
class CandidateDataset:
    """Metadata for a versioned candidate dataset."""

    version: str
    sample_count: int
    filtered_count: int
    output_manifest: Path


FEATURE_AE_GOOD_V002 = "feature_ae_good_v002"
FEATURE_AE_GOOD_V003 = "feature_ae_good_v003"


def _is_good_label(sample: CastingImageSample) -> bool:
    """Check if sample has a good label."""
    label = sample.label.lower()
    return label in {"good", "normal", "conforme"}


def _is_roi_ok(sample: CastingImageSample, roi_status: dict[str, str] | None = None) -> bool:
    """Check if sample has ROI OK status."""
    if not roi_status:
        return True
    status = roi_status.get(sample.image_id) or roi_status.get(sample.relative_path)
    if not status:
        return True
    return status.lower() == "ok"


def _is_not_defective(sample: CastingImageSample) -> bool:
    """Check if sample has no defects."""
    return not sample.is_defective


def _not_in_validation_set(sample: CastingImageSample) -> bool:
    """Check if sample is not in validation set."""
    split = sample.split_set.lower()
    return VALIDATION_SET_ID not in split


def _is_oracle_train_eligible(sample: CastingImageSample) -> bool:
    return (
        sample.oracle_verdict == "conforme"
        and sample.train_eligible is True
        and sample.train_eligibility_source == "oracle_gt"
        and not sample.quarantine_reason
    )


def filter_candidate_samples(
    samples: Iterable[CastingImageSample],
    *,
    roi_status: dict[str, str] | None = None,
) -> list[CastingImageSample]:
    """Filter samples with AI safety rules.

    Applies the following filters in order:
    - good only (label in {good, normal, conforme})
    - ROI OK (roi_status == "ok" if available)
    - no defects (is_defective == False)
    - exclude validation_set (split_set not containing "validation_set")
    """
    filtered = []
    for sample in samples:
        if not _is_good_label(sample):
            continue
        if not _is_roi_ok(sample, roi_status):
            continue
        if not _is_not_defective(sample):
            continue
        if not _not_in_validation_set(sample):
            continue
        filtered.append(sample)
    return filtered


def write_candidate_manifest(
    samples: Iterable[CastingImageSample],
    output_path: Path,
    version: str | None = None,
    *,
    manifest_version: str | None = None,
) -> None:
    """Write candidate samples to a CSV manifest.

    The output manifest follows the IQA casting dataset format with standard
    columns for image identification, classification, and quality metadata.

    Args:
        samples: Samples to write.
        output_path: Path to write the manifest CSV.
        version: Optional version to replace dataset_version in output.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples_list = list(samples)

    fieldnames = [
        "image_id",
        "image_ids",
        "relative_path",
        "relative_paths",
        "event_id",
        "source_class",
        "split_set",
        "label",
        "is_defective",
        "scenario_id",
        "dataset_version",
        "manifest_version",
        "gt_mask_path",
        "oracle_verdict",
        "train_eligible",
        "train_eligibility_source",
        "quarantine_reason",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples_list:
            writer.writerow({
                "image_id": sample.image_id,
                "image_ids": sample.image_id,
                "relative_path": sample.relative_path,
                "relative_paths": sample.relative_path,
                "event_id": sample.event_id,
                "source_class": sample.source_class,
                "split_set": sample.split_set,
                "label": sample.label,
                "is_defective": sample.is_defective,
                "scenario_id": sample.scenario_id,
                "dataset_version": version or sample.dataset_version,
                "manifest_version": manifest_version or version or sample.dataset_version,
                "gt_mask_path": sample.gt_mask_path,
                "oracle_verdict": sample.oracle_verdict,
                "train_eligible": str(sample.train_eligible).lower(),
                "train_eligibility_source": sample.train_eligibility_source,
                "quarantine_reason": sample.quarantine_reason,
            })


def build_candidate_dataset(
    samples: Iterable[CastingImageSample],
    output_manifest: Path,
    version: str = "v001",
    *,
    roi_status: dict[str, str] | None = None,
    manifest_version: str | None = None,
) -> CandidateDataset:
    """Build a versioned candidate dataset with safety filters.

    Args:
        samples: Input samples to filter.
        output_manifest: Path to write the candidate manifest.
        version: Dataset version identifier (e.g., "candidate_v001").
        roi_status: Optional dict mapping image_id/relative_path to ROI status.

    Returns:
        CandidateDataset metadata with version, counts, and output path.
    """
    samples_list = list(samples)
    initial_count = len(samples_list)
    filtered = filter_candidate_samples(samples_list, roi_status=roi_status)
    filtered_count = initial_count - len(filtered)

    write_candidate_manifest(filtered, output_manifest, version=version, manifest_version=manifest_version)

    return CandidateDataset(
        version=version,
        sample_count=len(filtered),
        filtered_count=filtered_count,
        output_manifest=output_manifest,
    )


def build_oracle_validated_feature_ae_dataset(
    samples: Iterable[CastingImageSample],
    output_manifest: Path,
    version: str,
    *,
    roi_status: dict[str, str] | None = None,
    manifest_version: str | None = None,
) -> CandidateDataset:
    """Build Feature-AE good-only datasets from oracle GT conforming samples."""

    samples_list = list(samples)
    base_filtered = filter_candidate_samples(samples_list, roi_status=roi_status)
    oracle_filtered = [sample for sample in base_filtered if _is_oracle_train_eligible(sample)]

    write_candidate_manifest(oracle_filtered, output_manifest, version=version, manifest_version=manifest_version)

    return CandidateDataset(
        version=version,
        sample_count=len(oracle_filtered),
        filtered_count=len(samples_list) - len(oracle_filtered),
        output_manifest=output_manifest,
    )


__all__ = [
    "CandidateDataset",
    "FEATURE_AE_GOOD_V002",
    "FEATURE_AE_GOOD_V003",
    "build_candidate_dataset",
    "build_oracle_validated_feature_ae_dataset",
    "filter_candidate_samples",
    "write_candidate_manifest",
]
