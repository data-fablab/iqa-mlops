"""Tests for the reconstruction-threshold calibration logic (Issue 6).

Pure derivation (class1 percentile + margin) and the YAML loader (missing file
tolerated). The GPU scoring itself is validated separately by verify.
"""

from __future__ import annotations

import pytest

from iqa.inference.reconstruction_calibration import (
    derive_reconstruction_thresholds,
    load_reconstruction_calibration,
)

pytestmark = pytest.mark.unit


class TestDeriveThresholds:
    def test_orange_below_red_and_above_typical_class1(self) -> None:
        scores = [0.01 * i for i in range(101)]  # 0.00 .. 1.00
        result = derive_reconstruction_thresholds(scores, orange_percentile=95.0, red_percentile=99.0, margin=0.0)
        assert result["threshold_orange"] == pytest.approx(0.95, abs=1e-6)
        assert result["threshold_red"] == pytest.approx(0.99, abs=1e-6)
        assert result["threshold_orange"] < result["threshold_red"]

    def test_margin_scales_thresholds_up(self) -> None:
        scores = [0.0, 0.5, 1.0]
        no_margin = derive_reconstruction_thresholds(scores, margin=0.0)
        with_margin = derive_reconstruction_thresholds(scores, margin=0.10)
        assert with_margin["threshold_orange"] == pytest.approx(no_margin["threshold_orange"] * 1.10)
        assert with_margin["threshold_red"] == pytest.approx(no_margin["threshold_red"] * 1.10)

    def test_red_never_below_orange_when_percentiles_collapse(self) -> None:
        # When orange and red percentiles land on the same value, red == orange.
        result = derive_reconstruction_thresholds([0.3] * 10, orange_percentile=95.0, red_percentile=99.0, margin=0.0)
        assert result["threshold_red"] >= result["threshold_orange"]

    def test_records_provenance_stats(self) -> None:
        result = derive_reconstruction_thresholds([0.1, 0.2, 0.3])
        stats = result["class1_score_stats"]
        assert stats["count"] == 3
        assert stats["min"] == pytest.approx(0.1)
        assert stats["max"] == pytest.approx(0.3)

    def test_empty_sample_raises(self) -> None:
        with pytest.raises(ValueError, match="empty class1 sample"):
            derive_reconstruction_thresholds([])

    def test_invalid_percentile_order_raises(self) -> None:
        with pytest.raises(ValueError, match="orange_percentile"):
            derive_reconstruction_thresholds([0.1, 0.2], orange_percentile=99.0, red_percentile=95.0)


class TestLoadCalibration:
    def test_missing_file_returns_none(self, tmp_path) -> None:
        assert load_reconstruction_calibration(tmp_path / "nope.yaml") is None

    def test_loads_thresholds_and_hitl_flag(self, tmp_path) -> None:
        path = tmp_path / "calib.yaml"
        path.write_text(
            "reconstruction_calibration:\n"
            "  thresholds:\n"
            "    threshold_orange: 0.31\n"
            "    threshold_red: 0.47\n"
            "  hitl:\n"
            "    validated: true\n",
            encoding="utf-8",
        )
        calib = load_reconstruction_calibration(path)
        assert calib is not None
        assert calib.threshold_orange == 0.31
        assert calib.threshold_red == 0.47
        assert calib.hitl_validated is True

    def test_incomplete_thresholds_returns_none(self, tmp_path) -> None:
        path = tmp_path / "calib.yaml"
        path.write_text(
            "reconstruction_calibration:\n  thresholds:\n    threshold_orange: 0.31\n",
            encoding="utf-8",
        )
        assert load_reconstruction_calibration(path) is None
