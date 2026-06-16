"""Tests for defect_coverage gate calculation and evaluation."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import yaml

from factories import make_sample
from iqa.datasets import CastingImageSample
from iqa.promotion import (
    compute_defect_coverage,
    check_defect_coverage_gate,
    compute_defect_coverage_from_manifest,
)


def _sample(
    source_class: str = "class_A",
    is_defective: bool = False,
) -> CastingImageSample:
    """Create a test sample keyed by source_class (defect coverage groups by class)."""
    return make_sample(
        image_id=f"img_{source_class}",
        source_class=source_class,
        is_defective=is_defective,
    )


class TestComputeDefectCoverage:
    """Test defect_coverage calculation."""

    def test_all_classes_have_defects_returns_100_percent(self) -> None:
        """When all source_classes have at least one defect, coverage is 1.0."""
        samples = [
            _sample(source_class="class_A", is_defective=True),
            _sample(source_class="class_B", is_defective=True),
            _sample(source_class="class_C", is_defective=True),
        ]
        coverage = compute_defect_coverage(samples)
        assert coverage == 1.0

    def test_partial_coverage_only_some_classes_have_defects(self) -> None:
        """When some source_classes lack defects, coverage is < 1.0."""
        samples = [
            _sample(source_class="class_A", is_defective=True),
            _sample(source_class="class_B", is_defective=False),  # no defect
            _sample(source_class="class_C", is_defective=False),  # no defect
        ]
        coverage = compute_defect_coverage(samples)
        assert coverage == 1.0 / 3.0

    def test_zero_coverage_no_defects_exist(self) -> None:
        """When no samples are defective, coverage is 0.0."""
        samples = [
            _sample(source_class="class_A", is_defective=False),
            _sample(source_class="class_B", is_defective=False),
        ]
        coverage = compute_defect_coverage(samples)
        assert coverage == 0.0

    def test_empty_sample_list_returns_zero(self) -> None:
        """Empty sample list returns 0.0 coverage."""
        coverage = compute_defect_coverage([])
        assert coverage == 0.0

    def test_multiple_defects_per_class_counts_once(self) -> None:
        """Multiple defective samples of same class count only once."""
        samples = [
            _sample(source_class="class_A", is_defective=True),
            _sample(source_class="class_A", is_defective=True),  # duplicate class
            _sample(source_class="class_B", is_defective=False),
        ]
        coverage = compute_defect_coverage(samples)
        assert coverage == 1.0 / 2.0  # 1 class with defects out of 2


class TestCheckDefectCoverageGate:
    """Test defect_coverage gate decision logic."""

    def test_gate_blocks_insufficient_coverage(self) -> None:
        """Gate blocks when coverage < min_coverage."""
        result = check_defect_coverage_gate(coverage=0.90, min_coverage=0.95)
        assert result["passed"] is False
        assert result["coverage"] == 0.90
        assert result["threshold"] == 0.95

    def test_gate_passes_sufficient_coverage(self) -> None:
        """Gate passes when coverage >= min_coverage."""
        result = check_defect_coverage_gate(coverage=0.95, min_coverage=0.95)
        assert result["passed"] is True
        assert result["coverage"] == 0.95
        assert result["threshold"] == 0.95

    def test_gate_passes_exceeding_coverage(self) -> None:
        """Gate passes when coverage > min_coverage."""
        result = check_defect_coverage_gate(coverage=1.0, min_coverage=0.95)
        assert result["passed"] is True
        assert result["coverage"] == 1.0

    def test_gate_blocks_zero_coverage(self) -> None:
        """Gate blocks when coverage is 0."""
        result = check_defect_coverage_gate(coverage=0.0, min_coverage=0.95)
        assert result["passed"] is False

    def test_gate_uses_default_threshold_095(self) -> None:
        """Gate uses 0.95 as default min_coverage."""
        result = check_defect_coverage_gate(coverage=0.95)
        assert result["threshold"] == 0.95
        assert result["passed"] is True


class TestDefectCoverageGateIntegration:
    """Integration tests with gate configuration."""

    def test_gate_with_config_threshold_blocks_insufficient_coverage(self) -> None:
        """Gate blocks insufficient coverage using config threshold."""
        gates_config = yaml.safe_load(Path("configs/promotion_gates.yaml").read_text(encoding="utf-8"))
        min_coverage = gates_config["defect_coverage"]["min_coverage"]

        # Coverage just below threshold
        coverage = min_coverage - 0.01
        result = check_defect_coverage_gate(coverage=coverage, min_coverage=min_coverage)
        assert result["passed"] is False

    def test_gate_with_config_threshold_passes_sufficient_coverage(self) -> None:
        """Gate passes when coverage meets config threshold."""
        gates_config = yaml.safe_load(Path("configs/promotion_gates.yaml").read_text(encoding="utf-8"))
        min_coverage = gates_config["defect_coverage"]["min_coverage"]

        # Coverage at exact threshold
        coverage = min_coverage
        result = check_defect_coverage_gate(coverage=coverage, min_coverage=min_coverage)
        assert result["passed"] is True

    def test_insufficient_defect_coverage_blocks_promotion(self) -> None:
        """End-to-end: insufficient defect coverage results in gate block."""
        gates_config = yaml.safe_load(Path("configs/promotion_gates.yaml").read_text(encoding="utf-8"))
        min_coverage = gates_config["defect_coverage"]["min_coverage"]

        samples = [
            _sample(source_class="class_A", is_defective=True),
            _sample(source_class="class_B", is_defective=False),
            _sample(source_class="class_C", is_defective=False),
            _sample(source_class="class_D", is_defective=False),
        ]

        coverage = compute_defect_coverage(samples)
        result = check_defect_coverage_gate(coverage=coverage, min_coverage=min_coverage)

        assert coverage == 0.25  # Only 1 out of 4 classes have defects
        assert result["passed"] is False


class TestDefectCoverageFromManifest:
    """Tests for manifest-based defect coverage calculation."""

    def test_compute_from_manifest_with_defective_samples(self) -> None:
        """Compute defect_coverage from a CSV manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.csv"
            with manifest_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["image_id", "relative_path", "source_class", "is_defective"],
                )
                writer.writeheader()
                writer.writerow({"image_id": "img_1", "relative_path": "path/1.jpg", "source_class": "class_A", "is_defective": "true"})
                writer.writerow({"image_id": "img_2", "relative_path": "path/2.jpg", "source_class": "class_B", "is_defective": "false"})
                writer.writerow({"image_id": "img_3", "relative_path": "path/3.jpg", "source_class": "class_C", "is_defective": "false"})

            coverage = compute_defect_coverage_from_manifest(manifest_path)
            assert coverage == 1.0 / 3.0  # 1 class with defects out of 3

    def test_compute_from_manifest_nonexistent_returns_zero(self) -> None:
        """Manifest file that doesn't exist returns 0.0."""
        coverage = compute_defect_coverage_from_manifest(Path("/nonexistent/manifest.csv"))
        assert coverage == 0.0

    def test_compute_from_manifest_all_defective(self) -> None:
        """All classes have defective samples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.csv"
            with manifest_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["image_id", "source_class", "is_defective"],
                )
                writer.writeheader()
                writer.writerow({"image_id": "img_1", "source_class": "class_A", "is_defective": "1"})
                writer.writerow({"image_id": "img_2", "source_class": "class_B", "is_defective": "YES"})
                writer.writerow({"image_id": "img_3", "source_class": "class_C", "is_defective": "defective"})

            coverage = compute_defect_coverage_from_manifest(manifest_path)
            assert coverage == 1.0
