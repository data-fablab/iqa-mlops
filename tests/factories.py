"""Shared test factories.

Centralises construction of domain objects used across multiple test modules so
the same defaults live in one place. Behaviour-defining values are passed as
keyword overrides by each test.
"""

from __future__ import annotations

from iqa.datasets import CastingImageSample


def make_sample(
    image_id: str = "img_001",
    event_id: str = "piece_event_001",
    label: str = "good",
    is_defective: bool = False,
    split_set: str = "train",
    source_class: str = "class_A",
    relative_path: str = "path/to/img.jpg",
    scenario_id: str = "scenario_1",
    dataset_version: str = "v001",
    gt_mask_path: str = "",
    oracle_verdict: str = "conforme",
    train_eligible: bool = True,
    train_eligibility_source: str = "oracle_gt",
    quarantine_reason: str = "",
) -> CastingImageSample:
    """Create a CastingImageSample with safe, training-eligible defaults."""
    return CastingImageSample(
        image_id=image_id,
        event_id=event_id,
        label=label,
        is_defective=is_defective,
        split_set=split_set,
        source_class=source_class,
        relative_path=relative_path,
        scenario_id=scenario_id,
        dataset_version=dataset_version,
        gt_mask_path=gt_mask_path,
        oracle_verdict=oracle_verdict,
        train_eligible=train_eligible,
        train_eligibility_source=train_eligibility_source,
        quarantine_reason=quarantine_reason,
    )
