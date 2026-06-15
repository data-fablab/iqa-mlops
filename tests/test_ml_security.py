"""Tests for ML security rules: training data validation and promotion gate blocking."""

from __future__ import annotations

import pytest

from iqa.datasets import (
    VALIDATION_SET_ID,
    CastingImageSample,
    build_candidate_dataset,
    filter_candidate_samples,
)
from iqa.promotion.gates import evaluate_promotion_gates


def _sample(
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
) -> CastingImageSample:
    """Create a test sample with default values."""
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
    )


class TestNoDefectivePiecesInTraining:
    """Verify defective pieces are excluded from training data."""

    def test_defective_piece_filtered_from_candidate(self) -> None:
        """Defective pieces must not appear in candidate dataset."""
        defective_sample = _sample(image_id="defective_piece", is_defective=True)
        good_sample = _sample(image_id="good_piece", is_defective=False)

        filtered = filter_candidate_samples([defective_sample, good_sample])

        assert len(filtered) == 1
        assert filtered[0].image_id == "good_piece"

    def test_all_defective_pieces_rejected(self) -> None:
        """All defective pieces must be filtered regardless of quantity."""
        samples = [
            _sample(image_id=f"defective_{i}", is_defective=True)
            for i in range(10)
        ]
        filtered = filter_candidate_samples(samples)
        assert len(filtered) == 0

    def test_candidate_dataset_contains_no_defective_pieces(self, tmp_path) -> None:
        """Built candidate dataset manifest must not list defective pieces."""
        output = tmp_path / "candidate.csv"
        samples = [
            _sample(image_id="good_1", is_defective=False),
            _sample(image_id="defective_1", is_defective=True),
            _sample(image_id="good_2", is_defective=False),
        ]

        result = build_candidate_dataset(samples, output)

        assert result.sample_count == 2
        content = output.read_text(encoding="utf-8")
        assert "defective_1" not in content
        assert "good_1" in content
        assert "good_2" in content


class TestNoValidationSetDataInTraining:
    """Verify validation set samples are excluded from training."""

    def test_validation_set_sample_filtered(self) -> None:
        """Validation set samples must be excluded from filtering."""
        validation_sample = _sample(
            image_id="validation_sample",
            split_set=VALIDATION_SET_ID,
        )
        train_sample = _sample(image_id="train_sample", split_set="train")

        filtered = filter_candidate_samples([validation_sample, train_sample])

        assert len(filtered) == 1
        assert filtered[0].image_id == "train_sample"

    def test_all_validation_set_excluded(self) -> None:
        """All validation set samples must be excluded."""
        samples = [
            _sample(image_id=f"val_{i}", split_set=VALIDATION_SET_ID)
            for i in range(5)
        ]
        filtered = filter_candidate_samples(samples)
        assert len(filtered) == 0

    def test_candidate_dataset_excludes_validation_set(self, tmp_path) -> None:
        """Candidate dataset must not include validation set samples."""
        output = tmp_path / "candidate.csv"
        samples = [
            _sample(image_id="train_1", split_set="train"),
            _sample(image_id="validation_1", split_set=VALIDATION_SET_ID),
            _sample(image_id="train_2", split_set="train"),
        ]

        result = build_candidate_dataset(samples, output)

        assert result.sample_count == 2
        content = output.read_text(encoding="utf-8")
        assert "validation_1" not in content
        assert "train_1" in content
        assert "train_2" in content


class TestROIFailExcludedFromTraining:
    """Verify ROI fail/warning samples are excluded from training."""

    def test_roi_fail_filtered(self) -> None:
        """ROI fail status must exclude sample from training."""
        fail_sample = _sample(image_id="roi_fail")
        ok_sample = _sample(image_id="roi_ok")

        roi_status = {"roi_fail": "fail", "roi_ok": "ok"}
        filtered = filter_candidate_samples([fail_sample, ok_sample], roi_status=roi_status)

        assert len(filtered) == 1
        assert filtered[0].image_id == "roi_ok"

    def test_roi_warning_filtered(self) -> None:
        """ROI warning status must exclude sample from training."""
        warning_sample = _sample(image_id="roi_warning")
        ok_sample = _sample(image_id="roi_ok")

        roi_status = {"roi_warning": "warning", "roi_ok": "ok"}
        filtered = filter_candidate_samples([warning_sample, ok_sample], roi_status=roi_status)

        assert len(filtered) == 1
        assert filtered[0].image_id == "roi_ok"

    def test_all_roi_failures_excluded(self) -> None:
        """All ROI non-ok statuses must be excluded."""
        samples = [
            _sample(image_id="fail_1"),
            _sample(image_id="fail_2"),
            _sample(image_id="warning_1"),
        ]
        roi_status = {
            "fail_1": "fail",
            "fail_2": "fail",
            "warning_1": "warning",
        }
        filtered = filter_candidate_samples(samples, roi_status=roi_status)
        assert len(filtered) == 0

    def test_candidate_dataset_excludes_roi_failures(self, tmp_path) -> None:
        """Candidate dataset must not include ROI fail/warning samples."""
        output = tmp_path / "candidate.csv"
        samples = [
            _sample(image_id="roi_ok_1"),
            _sample(image_id="roi_fail_1"),
            _sample(image_id="roi_warning_1"),
            _sample(image_id="roi_ok_2"),
        ]
        roi_status = {
            "roi_ok_1": "ok",
            "roi_fail_1": "fail",
            "roi_warning_1": "warning",
            "roi_ok_2": "ok",
        }

        result = build_candidate_dataset(samples, output, roi_status=roi_status)

        assert result.sample_count == 2
        content = output.read_text(encoding="utf-8")
        assert "roi_fail_1" not in content
        assert "roi_warning_1" not in content
        assert "roi_ok_1" in content
        assert "roi_ok_2" in content


class TestOrangeRateGateBlocking:
    """Verify orange rate gate blocks models with excessive orange predictions."""

    def test_orange_rate_gate_blocks_at_threshold(self) -> None:
        """Orange rate at max_rate threshold must not block."""
        result = evaluate_promotion_gates(
            candidate_recall=1.0,
            candidate_ap=0.95,
            candidate_orange_rate=0.10,  # At threshold
            candidate_latency_ms=900.0,
            prod_ap=0.95,
            gates_config={
                "feature_ae": {
                    "recall_defect_min": 1.0,
                    "image_ap_max_regression": 0.02,
                    "orange_rate_max": 0.10,
                    "latency_p95_ms_max": 1000.0,
                }
            },
        )
        assert result["all_passed"] is True
        assert result["gates"]["orange_rate"]["passed"] is True

    def test_orange_rate_gate_blocks_exceeding_threshold(self) -> None:
        """Orange rate exceeding max_rate must block promotion."""
        result = evaluate_promotion_gates(
            candidate_recall=1.0,
            candidate_ap=0.95,
            candidate_orange_rate=0.15,  # Exceeds threshold
            candidate_latency_ms=900.0,
            prod_ap=0.95,
            gates_config={
                "feature_ae": {
                    "recall_defect_min": 1.0,
                    "image_ap_max_regression": 0.02,
                    "orange_rate_max": 0.10,
                    "latency_p95_ms_max": 1000.0,
                }
            },
        )
        assert result["all_passed"] is False
        assert result["rollback_signal"] is True
        assert result["gates"]["orange_rate"]["passed"] is False

    def test_orange_rate_exceeding_triggers_rollback(self) -> None:
        """Exceeding orange rate must trigger rollback signal."""
        result = evaluate_promotion_gates(
            candidate_recall=1.0,
            candidate_ap=0.94,
            candidate_orange_rate=0.20,
            candidate_latency_ms=900.0,
            prod_ap=0.95,
            gates_config={
                "feature_ae": {
                    "recall_defect_min": 1.0,
                    "image_ap_max_regression": 0.02,
                    "orange_rate_max": 0.10,
                    "latency_p95_ms_max": 1000.0,
                }
            },
        )
        assert result["rollback_signal"] is True


class TestGatesBlockPromotionOnFailure:
    """Verify blocking gates prevent unsafe model promotion."""

    def test_multiple_gate_failures_block_promotion(self) -> None:
        """Multiple gate failures must block promotion."""
        result = evaluate_promotion_gates(
            candidate_recall=0.95,  # Below threshold
            candidate_ap=0.92,  # Regresses
            candidate_orange_rate=0.15,  # Too high
            candidate_latency_ms=1100.0,  # Too slow
            prod_ap=0.95,
            gates_config={
                "feature_ae": {
                    "recall_defect_min": 1.0,
                    "image_ap_max_regression": 0.02,
                    "orange_rate_max": 0.10,
                    "latency_p95_ms_max": 1000.0,
                }
            },
        )
        assert result["all_passed"] is False
        assert result["rollback_signal"] is True
        # Verify multiple gates failed
        failed_gates = [g for g in result["gates"].values() if not g["passed"]]
        assert len(failed_gates) >= 3

    def test_single_gate_failure_blocks_entire_promotion(self) -> None:
        """Single gate failure must block even if others pass."""
        result = evaluate_promotion_gates(
            candidate_recall=1.0,  # Pass
            candidate_ap=0.94,  # Pass
            candidate_orange_rate=0.20,  # FAIL
            candidate_latency_ms=900.0,  # Pass
            prod_ap=0.95,
            gates_config={
                "feature_ae": {
                    "recall_defect_min": 1.0,
                    "image_ap_max_regression": 0.02,
                    "orange_rate_max": 0.10,
                    "latency_p95_ms_max": 1000.0,
                }
            },
        )
        assert result["all_passed"] is False
        assert result["rollback_signal"] is True


class TestTrainingDataSecurityPipeline:
    """Integration tests for complete training data security validation."""

    def test_candidate_dataset_applies_all_safety_rules(self, tmp_path) -> None:
        """Candidate dataset applies all safety rules: defects, validation_set, ROI."""
        output = tmp_path / "candidate.csv"
        samples = [
            _sample(image_id="good_safe", label="good", is_defective=False, split_set="train"),
            _sample(image_id="bad_label", label="defect", split_set="train"),
            _sample(image_id="defective_piece", is_defective=True, split_set="train"),
            _sample(image_id="from_validation", split_set=VALIDATION_SET_ID),
            _sample(image_id="roi_fail", split_set="train"),
        ]
        roi_status = {"roi_fail": "fail"}

        result = build_candidate_dataset(samples, output, roi_status=roi_status)

        assert result.sample_count == 1
        assert result.filtered_count == 4
        content = output.read_text(encoding="utf-8")
        assert "good_safe" in content
        assert "bad_label" not in content
        assert "defective_piece" not in content
        assert "from_validation" not in content
        assert "roi_fail" not in content

    def test_mixed_good_and_bad_samples_filters_correctly(self, tmp_path) -> None:
        """Mixed dataset with good and bad samples filters correctly."""
        output = tmp_path / "candidate.csv"
        samples = [
            _sample(image_id=f"sample_{i}", label="good", is_defective=False)
            for i in range(5)
        ] + [
            _sample(image_id=f"bad_{i}", label="bad", is_defective=True)
            for i in range(3)
        ]

        result = build_candidate_dataset(samples, output)

        assert result.sample_count == 5
        assert result.filtered_count == 3


__all__ = [
    "TestNoDefectivePiecesInTraining",
    "TestNoValidationSetDataInTraining",
    "TestROIFailExcludedFromTraining",
    "TestOrangeRateGateBlocking",
    "TestGatesBlockPromotionOnFailure",
    "TestTrainingDataSecurityPipeline",
]
