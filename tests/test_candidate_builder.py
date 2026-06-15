"""Tests for candidate dataset builder with AI safety filters."""

from __future__ import annotations

from pathlib import Path

from iqa.datasets import (
    VALIDATION_SET_ID,
    CastingImageSample,
    build_candidate_dataset,
    filter_candidate_samples,
    write_candidate_manifest,
)


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


class TestFilterCandidateSamples:
    """Test filtering logic for candidate dataset safety rules."""

    def test_filter_good_only_accepts_good_label(self) -> None:
        """Filter accepts samples with 'good' label."""
        sample = _sample(label="good")
        filtered = filter_candidate_samples([sample])
        assert len(filtered) == 1

    def test_filter_good_only_accepts_normal_label(self) -> None:
        """Filter accepts samples with 'normal' label."""
        sample = _sample(label="normal")
        filtered = filter_candidate_samples([sample])
        assert len(filtered) == 1

    def test_filter_good_only_accepts_conforme_label(self) -> None:
        """Filter accepts samples with 'conforme' label."""
        sample = _sample(label="conforme")
        filtered = filter_candidate_samples([sample])
        assert len(filtered) == 1

    def test_filter_good_only_rejects_defect_label(self) -> None:
        """Filter rejects samples with defect labels."""
        sample = _sample(label="defect")
        filtered = filter_candidate_samples([sample])
        assert len(filtered) == 0

    def test_filter_good_only_rejects_bad_label(self) -> None:
        """Filter rejects samples with 'bad' label."""
        sample = _sample(label="bad")
        filtered = filter_candidate_samples([sample])
        assert len(filtered) == 0

    def test_filter_no_defects_accepts_non_defective(self) -> None:
        """Filter accepts samples with is_defective=False."""
        sample = _sample(is_defective=False)
        filtered = filter_candidate_samples([sample])
        assert len(filtered) == 1

    def test_filter_no_defects_rejects_defective(self) -> None:
        """Filter rejects samples with is_defective=True."""
        sample = _sample(is_defective=True)
        filtered = filter_candidate_samples([sample])
        assert len(filtered) == 0

    def test_filter_excludes_validation_set(self) -> None:
        """Filter rejects samples from validation_set."""
        sample = _sample(split_set="validation_set_v001")
        filtered = filter_candidate_samples([sample])
        assert len(filtered) == 0

    def test_filter_accepts_other_splits(self) -> None:
        """Filter accepts samples from non-validation splits."""
        sample = _sample(split_set="train")
        filtered = filter_candidate_samples([sample])
        assert len(filtered) == 1

    def test_filter_roi_ok_accepts_when_no_roi_status(self) -> None:
        """Filter accepts samples when no ROI status provided."""
        sample = _sample()
        filtered = filter_candidate_samples([sample], roi_status=None)
        assert len(filtered) == 1

    def test_filter_roi_ok_accepts_when_status_ok(self) -> None:
        """Filter accepts samples with roi_status='ok'."""
        sample = _sample(image_id="img_001")
        roi_status = {"img_001": "ok"}
        filtered = filter_candidate_samples([sample], roi_status=roi_status)
        assert len(filtered) == 1

    def test_filter_roi_ok_accepts_when_status_not_provided(self) -> None:
        """Filter accepts samples when ROI status key not in dict."""
        sample = _sample(image_id="img_001")
        roi_status = {"img_002": "ok"}
        filtered = filter_candidate_samples([sample], roi_status=roi_status)
        assert len(filtered) == 1

    def test_filter_roi_ok_rejects_when_status_not_ok(self) -> None:
        """Filter rejects samples with roi_status != 'ok'."""
        sample = _sample(image_id="img_001")
        roi_status = {"img_001": "failed"}
        filtered = filter_candidate_samples([sample], roi_status=roi_status)
        assert len(filtered) == 0

    def test_filter_roi_ok_case_insensitive(self) -> None:
        """Filter is case-insensitive for ROI status."""
        sample = _sample(image_id="img_001")
        roi_status = {"img_001": "OK"}
        filtered = filter_candidate_samples([sample], roi_status=roi_status)
        assert len(filtered) == 1

    def test_filter_roi_ok_by_relative_path(self) -> None:
        """Filter matches ROI status by relative_path when image_id not found."""
        sample = _sample(image_id="unknown", relative_path="path/to/img.jpg")
        roi_status = {"path/to/img.jpg": "ok"}
        filtered = filter_candidate_samples([sample], roi_status=roi_status)
        assert len(filtered) == 1

    def test_filter_roi_rejects_warning(self) -> None:
        """Filter rejects samples with roi_status='warning'."""
        sample = _sample(image_id="img_001")
        roi_status = {"img_001": "warning"}
        filtered = filter_candidate_samples([sample], roi_status=roi_status)
        assert len(filtered) == 0

    def test_filter_roi_rejects_fail(self) -> None:
        """Filter rejects samples with roi_status='fail'."""
        sample = _sample(image_id="img_001")
        roi_status = {"img_001": "fail"}
        filtered = filter_candidate_samples([sample], roi_status=roi_status)
        assert len(filtered) == 0

    def test_filter_combines_all_rules(self) -> None:
        """Filter combines all safety rules in sequence."""
        samples = [
            _sample(image_id="good", label="good", is_defective=False),
            _sample(image_id="bad_label", label="defect"),
            _sample(image_id="defective", is_defective=True),
            _sample(image_id="validation", split_set=VALIDATION_SET_ID),
        ]
        filtered = filter_candidate_samples(samples)
        assert len(filtered) == 1
        assert filtered[0].image_id == "good"

    def test_filter_multiple_samples(self) -> None:
        """Filter processes multiple samples correctly."""
        samples = [
            _sample(image_id=f"img_{i:03d}", label="good" if i % 2 == 0 else "bad")
            for i in range(10)
        ]
        filtered = filter_candidate_samples(samples)
        assert len(filtered) == 5
        assert all(s.label == "good" for s in filtered)


class TestWriteCandidateManifest:
    """Test writing candidate dataset manifests."""

    def test_write_manifest_creates_file(self, tmp_path: Path) -> None:
        """Write manifest creates output file."""
        output = tmp_path / "manifest.csv"
        sample = _sample()
        write_candidate_manifest([sample], output)
        assert output.exists()

    def test_write_manifest_includes_headers(self, tmp_path: Path) -> None:
        """Write manifest includes CSV headers."""
        output = tmp_path / "manifest.csv"
        sample = _sample()
        write_candidate_manifest([sample], output)
        content = output.read_text(encoding="utf-8")
        assert "image_id" in content
        assert "event_id" in content
        assert "label" in content

    def test_write_manifest_includes_sample_data(self, tmp_path: Path) -> None:
        """Write manifest includes sample data."""
        output = tmp_path / "manifest.csv"
        sample = _sample(image_id="test_img", event_id="evt_001")
        write_candidate_manifest([sample], output)
        content = output.read_text(encoding="utf-8")
        assert "test_img" in content
        assert "evt_001" in content

    def test_write_manifest_creates_parent_directory(self, tmp_path: Path) -> None:
        """Write manifest creates parent directories as needed."""
        output = tmp_path / "subdir" / "manifest.csv"
        sample = _sample()
        write_candidate_manifest([sample], output)
        assert output.exists()
        assert output.parent.exists()

    def test_write_manifest_multiple_samples(self, tmp_path: Path) -> None:
        """Write manifest handles multiple samples."""
        output = tmp_path / "manifest.csv"
        samples = [_sample(image_id=f"img_{i}") for i in range(5)]
        write_candidate_manifest(samples, output)
        content = output.read_text(encoding="utf-8")
        for sample in samples:
            assert sample.image_id in content


class TestBuildCandidateDataset:
    """Test building candidate datasets with filtering and versioning."""

    def test_build_candidate_basic(self, tmp_path: Path) -> None:
        """Build candidate dataset creates versioned output."""
        output = tmp_path / "candidate.csv"
        sample = _sample(label="good", is_defective=False)
        result = build_candidate_dataset([sample], output, version="candidate_v001")
        assert result.version == "candidate_v001"
        assert result.sample_count == 1
        assert result.filtered_count == 0
        assert output.exists()

    def test_build_candidate_counts_filtered_samples(self, tmp_path: Path) -> None:
        """Build candidate dataset counts filtered samples."""
        output = tmp_path / "candidate.csv"
        samples = [
            _sample(image_id="good", label="good"),
            _sample(image_id="bad", label="defect"),
            _sample(image_id="defective", is_defective=True),
        ]
        result = build_candidate_dataset(samples, output)
        assert result.sample_count == 1
        assert result.filtered_count == 2

    def test_build_candidate_applies_safety_filters(self, tmp_path: Path) -> None:
        """Build candidate dataset applies all safety filters."""
        output = tmp_path / "candidate.csv"
        samples = [
            _sample(image_id="pass", label="good", is_defective=False),
            _sample(image_id="fail_label", label="bad"),
            _sample(image_id="fail_defect", is_defective=True),
            _sample(image_id="fail_validation", split_set=VALIDATION_SET_ID),
        ]
        result = build_candidate_dataset(samples, output)
        assert result.sample_count == 1
        assert result.filtered_count == 3
        content = output.read_text(encoding="utf-8")
        assert "pass" in content
        assert "fail_label" not in content

    def test_build_candidate_with_roi_status(self, tmp_path: Path) -> None:
        """Build candidate dataset applies ROI filter when provided."""
        output = tmp_path / "candidate.csv"
        samples = [
            _sample(image_id="roi_ok"),
            _sample(image_id="roi_fail"),
        ]
        roi_status = {"roi_ok": "ok", "roi_fail": "failed"}
        result = build_candidate_dataset(samples, output, roi_status=roi_status)
        assert result.sample_count == 1
        assert result.filtered_count == 1

    def test_build_candidate_version_in_manifest(self, tmp_path: Path) -> None:
        """Build candidate dataset includes version in output path."""
        output = tmp_path / "candidate.csv"
        sample = _sample()
        result = build_candidate_dataset([sample], output, version="candidate_v002")
        assert result.version == "candidate_v002"
        assert result.output_manifest == output

    def test_build_candidate_replaces_dataset_version(self, tmp_path: Path) -> None:
        """Build candidate dataset replaces dataset_version with candidate version."""
        output = tmp_path / "candidate.csv"
        sample = _sample(dataset_version="v001")
        result = build_candidate_dataset([sample], output, version="candidate_v005")

        content = output.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        data_line = lines[1]  # Skip header

        assert "candidate_v005" in content
        assert "v001" not in data_line  # Original version should be replaced

    def test_build_candidate_empty_input(self, tmp_path: Path) -> None:
        """Build candidate dataset handles empty input."""
        output = tmp_path / "candidate.csv"
        result = build_candidate_dataset([], output)
        assert result.sample_count == 0
        assert result.filtered_count == 0
        assert output.exists()


class TestCandidateBuilderIntegration:
    """Integration tests for the complete candidate building pipeline."""

    def test_candidate_dataset_pipeline(self, tmp_path: Path) -> None:
        """Complete pipeline: filter, write, and verify candidate dataset."""
        output = tmp_path / "candidates" / "dataset_v001.csv"
        samples = [
            _sample(
                image_id=f"img_{i:03d}",
                event_id=f"evt_{i:03d}",
                label="good" if i % 2 == 0 else "bad",
                is_defective=i % 3 == 0,
                split_set="train" if i % 5 != 0 else VALIDATION_SET_ID,
            )
            for i in range(10)
        ]

        result = build_candidate_dataset(samples, output, version="dataset_v001")

        assert result.sample_count > 0
        assert result.filtered_count == 10 - result.sample_count
        assert output.exists()

        content = output.read_text(encoding="utf-8")
        assert "image_id" in content

        for sample in samples:
            should_be_included = (
                sample.label.lower() in {"good", "normal", "conforme"}
                and not sample.is_defective
                and VALIDATION_SET_ID not in sample.split_set.lower()
            )
            if should_be_included:
                assert sample.image_id in content
