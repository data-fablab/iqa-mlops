"""End-to-end ML security invariants.

These tests assert the *security guarantees* of the training pipeline at the
integrated level (built candidate manifest + promotion blocking). The unit-level
filtering rules live in tests/test_candidate_builder.py and the per-gate logic in
tests/test_promotion_gates.py -- they are intentionally NOT re-tested here.
"""

from __future__ import annotations


from iqa.datasets import (
    VALIDATION_SET_ID,
    CastingImageSample,
    build_candidate_dataset,
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


_FEATURE_AE_GATES = {
    "feature_ae": {
        "recall_defect_min": 1.0,
        "image_ap_max_regression": 0.02,
        "orange_rate_max": 0.10,
        "latency_p95_ms_max": 1000.0,
    }
}


class TestTrainingManifestSecurityInvariants:
    """The built candidate manifest must never leak unsafe training samples."""

    def test_defective_pieces_never_reach_manifest(self, tmp_path) -> None:
        """Defective pieces must not appear in the built candidate manifest."""
        output = tmp_path / "candidate.csv"
        result = build_candidate_dataset(
            [
                _sample(image_id="good_1", is_defective=False),
                _sample(image_id="defective_1", is_defective=True),
            ],
            output,
        )
        assert result.sample_count == 1
        content = output.read_text(encoding="utf-8")
        assert "defective_1" not in content
        assert "good_1" in content

    def test_validation_set_never_reaches_manifest(self, tmp_path) -> None:
        """Validation set samples must not appear in the built candidate manifest."""
        output = tmp_path / "candidate.csv"
        result = build_candidate_dataset(
            [
                _sample(image_id="train_1", split_set="train"),
                _sample(image_id="validation_1", split_set=VALIDATION_SET_ID),
            ],
            output,
        )
        assert result.sample_count == 1
        content = output.read_text(encoding="utf-8")
        assert "validation_1" not in content
        assert "train_1" in content

    def test_roi_failures_never_reach_manifest(self, tmp_path) -> None:
        """ROI fail/warning samples must not appear in the built candidate manifest."""
        output = tmp_path / "candidate.csv"
        result = build_candidate_dataset(
            [
                _sample(image_id="roi_ok_1"),
                _sample(image_id="roi_fail_1"),
                _sample(image_id="roi_warning_1"),
            ],
            output,
            roi_status={
                "roi_ok_1": "ok",
                "roi_fail_1": "fail",
                "roi_warning_1": "warning",
            },
        )
        assert result.sample_count == 1
        content = output.read_text(encoding="utf-8")
        assert "roi_fail_1" not in content
        assert "roi_warning_1" not in content
        assert "roi_ok_1" in content

    def test_all_safety_rules_apply_together(self, tmp_path) -> None:
        """A single build pass enforces defect, validation_set and ROI rules at once."""
        output = tmp_path / "candidate.csv"
        samples = [
            _sample(image_id="good_safe", label="good", is_defective=False),
            _sample(image_id="bad_label", label="defect"),
            _sample(image_id="defective_piece", is_defective=True),
            _sample(image_id="from_validation", split_set=VALIDATION_SET_ID),
            _sample(image_id="roi_fail", split_set="train"),
        ]
        result = build_candidate_dataset(
            samples, output, roi_status={"roi_fail": "fail"}
        )
        assert result.sample_count == 1
        assert result.filtered_count == 4
        content = output.read_text(encoding="utf-8")
        assert "good_safe" in content
        for leaked in ("bad_label", "defective_piece", "from_validation", "roi_fail"):
            assert leaked not in content


class TestUnsafePromotionIsBlocked:
    """An unsafe candidate must be blocked and raise the rollback signal."""

    def test_unsafe_candidate_blocks_and_signals_rollback(self) -> None:
        """A candidate failing one or more gates blocks promotion and signals rollback."""
        result = evaluate_promotion_gates(
            candidate_recall=0.95,  # below threshold
            candidate_ap=0.92,  # regresses
            candidate_orange_rate=0.15,  # too high
            candidate_latency_ms=1100.0,  # too slow
            prod_ap=0.95,
            gates_config=_FEATURE_AE_GATES,
        )
        assert result["all_passed"] is False
        assert result["rollback_signal"] is True

    def test_safe_candidate_passes_all_gates(self) -> None:
        """A fully compliant candidate passes and does not signal rollback."""
        result = evaluate_promotion_gates(
            candidate_recall=1.0,
            candidate_ap=0.95,
            candidate_orange_rate=0.08,
            candidate_latency_ms=900.0,
            prod_ap=0.95,
            gates_config=_FEATURE_AE_GATES,
        )
        assert result["all_passed"] is True
        assert result["rollback_signal"] is False
