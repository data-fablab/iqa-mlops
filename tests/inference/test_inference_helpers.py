"""Tests for inference helper functions."""

from __future__ import annotations

import time

from iqa.inference.helpers import compute_status, measure_inference_time


class TestComputeStatus:
    """Test status computation from anomaly scores."""

    def test_green_below_orange_threshold(self) -> None:
        """Score below orange threshold is green."""
        status = compute_status(0.01, threshold_orange=0.02, threshold_red=0.05)
        assert status == "green"

    def test_orange_between_thresholds(self) -> None:
        """Score between thresholds is orange."""
        status = compute_status(0.03, threshold_orange=0.02, threshold_red=0.05)
        assert status == "orange"

    def test_red_above_red_threshold(self) -> None:
        """Score above red threshold is red."""
        status = compute_status(0.06, threshold_orange=0.02, threshold_red=0.05)
        assert status == "red"

    def test_orange_at_boundary(self) -> None:
        """Score exactly at orange threshold is orange (inclusive)."""
        status = compute_status(0.02, threshold_orange=0.02, threshold_red=0.05)
        assert status == "orange"

    def test_red_at_boundary(self) -> None:
        """Score exactly at red threshold is red (inclusive)."""
        status = compute_status(0.05, threshold_orange=0.02, threshold_red=0.05)
        assert status == "red"

    def test_zero_score_is_green(self) -> None:
        """Zero score (perfect) is green."""
        status = compute_status(0.0, threshold_orange=0.02, threshold_red=0.05)
        assert status == "green"

    def test_high_score_is_red(self) -> None:
        """Very high score is red."""
        status = compute_status(999.9, threshold_orange=0.02, threshold_red=0.05)
        assert status == "red"


class TestMeasureInferenceTime:
    """Test inference timing helper."""

    def test_measure_inference_time_captures_duration(self) -> None:
        """Context manager captures elapsed time."""
        with measure_inference_time() as timing:
            time.sleep(0.01)

        assert "elapsed_ms" in timing
        assert timing["elapsed_ms"] >= 10.0

    def test_measure_inference_time_accurate(self) -> None:
        """Elapsed time is approximately accurate."""
        with measure_inference_time() as timing:
            time.sleep(0.05)

        assert 40.0 < timing["elapsed_ms"] < 100.0

    def test_measure_inference_time_very_fast(self) -> None:
        """Can measure very fast operations."""
        with measure_inference_time() as timing:
            pass

        assert timing["elapsed_ms"] >= 0.0
        assert timing["elapsed_ms"] < 10.0

    def test_measure_inference_time_multiple_calls(self) -> None:
        """Multiple timing measurements are independent."""
        timings = []
        for _ in range(3):
            with measure_inference_time() as timing:
                time.sleep(0.01)
            timings.append(timing["elapsed_ms"])

        assert all(t >= 10.0 for t in timings)
        assert len(timings) == 3
