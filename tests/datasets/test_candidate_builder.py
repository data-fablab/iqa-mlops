"""Tests for candidate dataset builder with AI safety filters."""

from __future__ import annotations

from pathlib import Path

import pytest
from factories import make_sample as _sample

from iqa.datasets import (
    FEATURE_AE_GOOD_V002,
    FEATURE_AE_GOOD_V003,
    VALIDATION_SET_ID,
    build_oracle_validated_feature_ae_dataset,
    build_candidate_dataset,
    filter_candidate_samples,
    write_candidate_manifest,
)


class TestFilterCandidateSamples:
    """Test filtering logic for candidate dataset safety rules."""

    @pytest.mark.parametrize(
        "sample_kwargs, expected",
        [
            ({"label": "good"}, 1),
            ({"label": "normal"}, 1),
            ({"label": "conforme"}, 1),
            ({"label": "defect"}, 0),
            ({"label": "bad"}, 0),
            ({"is_defective": False}, 1),
            ({"is_defective": True}, 0),
            ({"split_set": "validation_set_v001"}, 0),
            ({"split_set": "train"}, 1),
        ],
        ids=[
            "label-good",
            "label-normal",
            "label-conforme",
            "label-defect",
            "label-bad",
            "not-defective",
            "is-defective",
            "validation-split",
            "train-split",
        ],
    )
    def test_filter_single_safety_rule(
        self, sample_kwargs: dict, expected: int
    ) -> None:
        """Each safety rule accepts good/non-defective/non-validation samples only."""
        filtered = filter_candidate_samples([_sample(**sample_kwargs)])
        assert len(filtered) == expected

    @pytest.mark.parametrize(
        "roi_status, expected",
        [
            (None, 1),
            ({"img_001": "ok"}, 1),
            ({"img_002": "ok"}, 1),
            ({"img_001": "OK"}, 1),
            ({"img_001": "failed"}, 0),
            ({"img_001": "warning"}, 0),
            ({"img_001": "fail"}, 0),
        ],
        ids=[
            "no-status",
            "status-ok",
            "key-absent",
            "ok-case-insensitive",
            "failed",
            "warning",
            "fail",
        ],
    )
    def test_filter_roi_status(self, roi_status: dict | None, expected: int) -> None:
        """ROI gate accepts only 'ok' (case-insensitive) or absent status."""
        sample = _sample(image_id="img_001")
        filtered = filter_candidate_samples([sample], roi_status=roi_status)
        assert len(filtered) == expected

    def test_filter_roi_ok_by_relative_path(self) -> None:
        """Filter matches ROI status by relative_path when image_id not found."""
        sample = _sample(image_id="unknown", relative_path="path/to/img.jpg")
        roi_status = {"path/to/img.jpg": "ok"}
        filtered = filter_candidate_samples([sample], roi_status=roi_status)
        assert len(filtered) == 1

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
        assert "manifest_version" in content

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
        build_candidate_dataset([sample], output, version="candidate_v005")

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


class TestBuildOracleValidatedFeatureAEDataset:
    """Test Feature-AE v002/v003 good-only oracle GT datasets."""

    def test_build_v002_keeps_only_oracle_conforme_train_eligible_samples(self, tmp_path: Path) -> None:
        output = tmp_path / "feature_ae_good_v002.csv"
        samples = [
            _sample(image_id="keep", oracle_verdict="conforme", train_eligible=True),
            _sample(image_id="defective", oracle_verdict="defective", train_eligible=False),
            _sample(image_id="human", train_eligibility_source="human_sophie"),
            _sample(image_id="quarantine", quarantine_reason="roi_fail"),
            _sample(image_id="defective_flag", is_defective=True),
        ]

        result = build_oracle_validated_feature_ae_dataset(samples, output, FEATURE_AE_GOOD_V002)
        content = output.read_text(encoding="utf-8")
        data_lines = content.splitlines()[1:]

        assert result.version == FEATURE_AE_GOOD_V002
        assert result.sample_count == 1
        assert result.filtered_count == 4
        assert "keep" in content
        assert all("defective" not in line for line in data_lines)
        assert all("human" not in line for line in data_lines)
        assert all("quarantine" not in line for line in data_lines)
        assert FEATURE_AE_GOOD_V002 in content

    def test_build_v003_uses_drift_dataset_version(self, tmp_path: Path) -> None:
        output = tmp_path / "feature_ae_good_v003.csv"
        sample = _sample(
            image_id="drift_conforme",
            scenario_id="drift_domain_extension",
            dataset_version="drift_domain_extension_v001",
            oracle_verdict="conforme",
            train_eligible=True,
            train_eligibility_source="oracle_gt",
        )

        result = build_oracle_validated_feature_ae_dataset([sample], output, FEATURE_AE_GOOD_V003)
        content = output.read_text(encoding="utf-8")

        assert result.version == FEATURE_AE_GOOD_V003
        assert result.sample_count == 1
        assert FEATURE_AE_GOOD_V003 in content
        assert "drift_domain_extension_v001" not in content.splitlines()[1]

    def test_build_oracle_dataset_writes_manifest_version(self, tmp_path: Path) -> None:
        output = tmp_path / "feature_ae_good_v002.csv"
        sample = _sample(image_id="versioned", oracle_verdict="conforme", train_eligible=True)

        build_oracle_validated_feature_ae_dataset(
            [sample],
            output,
            FEATURE_AE_GOOD_V002,
            manifest_version="feature_ae_good_v002_manifest_v001",
        )
        content = output.read_text(encoding="utf-8")

        assert "manifest_version" in content.splitlines()[0]
        assert "feature_ae_good_v002_manifest_v001" in content.splitlines()[1]


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
