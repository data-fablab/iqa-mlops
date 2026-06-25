"""Tests for the 4-metric non-regression promotion gate (Issue 4).

The gate is a *non-regression vs prod baseline* over the business metrics in
priority order ``pixel_aupimo_1e-5_1e-3 -> pixel_ap -> image_ap -> image_auroc``
(ADR 0010 §6), with a fallback to ``image_ap`` when GT masks are absent so the
pixel metrics cannot be computed.
"""

from __future__ import annotations

import pytest

from iqa.monitoring.model_metrics import AUPIMO_KEY
from iqa.promotion.gates import (
    evaluate_promotion_gates,
    evaluate_quality_regression_gates,
)


def _metrics(aupimo=0.80, pixel_ap=0.70, image_ap=0.90, image_auroc=0.95):
    return {
        AUPIMO_KEY: aupimo,
        "pixel_ap": pixel_ap,
        "image_ap": image_ap,
        "image_auroc": image_auroc,
    }


class TestQualityRegressionGate:
    def test_non_regression_passes_all_four_metrics(self) -> None:
        """A candidate at least as good as prod on every metric passes."""
        result = evaluate_quality_regression_gates(
            candidate_metrics=_metrics(0.81, 0.71, 0.91, 0.96),
            prod_metrics=_metrics(0.80, 0.70, 0.90, 0.95),
        )
        assert result["all_passed"] is True
        assert set(result["evaluated_metrics"]) == {
            AUPIMO_KEY,
            "pixel_ap",
            "image_ap",
            "image_auroc",
        }
        assert result["decisive_metric"] == AUPIMO_KEY
        assert result["fallback_to_image_ap"] is False

    def test_within_tolerance_passes(self) -> None:
        """A small drop within max_regression still passes."""
        result = evaluate_quality_regression_gates(
            candidate_metrics=_metrics(aupimo=0.79),  # 0.01 drop, default 0.02
            prod_metrics=_metrics(aupimo=0.80),
        )
        assert result["all_passed"] is True
        assert result["metrics"][AUPIMO_KEY]["regression"] == pytest.approx(0.01)
        assert result["metrics"][AUPIMO_KEY]["passed"] is True

    def test_regression_beyond_tolerance_blocks(self) -> None:
        """A drop larger than max_regression on the decisive metric blocks."""
        result = evaluate_quality_regression_gates(
            candidate_metrics=_metrics(aupimo=0.75),  # 0.05 drop > 0.02
            prod_metrics=_metrics(aupimo=0.80),
        )
        assert result["all_passed"] is False
        assert result["metrics"][AUPIMO_KEY]["passed"] is False
        assert result["decisive_metric"] == AUPIMO_KEY

    def test_per_metric_max_regression_from_config(self) -> None:
        """Per-metric thresholds override the default."""
        result = evaluate_quality_regression_gates(
            candidate_metrics=_metrics(aupimo=0.76),  # 0.04 drop
            prod_metrics=_metrics(aupimo=0.80),
            max_regressions={AUPIMO_KEY: 0.05},
        )
        assert result["metrics"][AUPIMO_KEY]["max_regression"] == 0.05
        assert result["all_passed"] is True

    def test_fallback_to_image_ap_when_masks_absent(self) -> None:
        """Without GT masks the pixel metrics are absent; gate falls back to image_ap."""
        candidate = {"image_ap": 0.92, "image_auroc": 0.96}
        prod = {"image_ap": 0.90, "image_auroc": 0.95}
        result = evaluate_quality_regression_gates(candidate, prod)
        assert result["fallback_to_image_ap"] is True
        assert result["decisive_metric"] == "image_ap"
        assert AUPIMO_KEY in result["skipped_metrics"]
        assert "pixel_ap" in result["skipped_metrics"]
        assert result["all_passed"] is True

    def test_fallback_image_ap_regression_blocks(self) -> None:
        """When falling back, an image_ap regression still blocks."""
        result = evaluate_quality_regression_gates(
            candidate_metrics={"image_ap": 0.85},
            prod_metrics={"image_ap": 0.90},  # 0.05 drop
        )
        assert result["fallback_to_image_ap"] is True
        assert result["decisive_metric"] == "image_ap"
        assert result["all_passed"] is False

    def test_priority_order_picks_highest_available_as_decisive(self) -> None:
        """With pixel_ap present but aupimo absent, pixel_ap is decisive."""
        candidate = {"pixel_ap": 0.71, "image_ap": 0.91}
        prod = {"pixel_ap": 0.70, "image_ap": 0.90}
        result = evaluate_quality_regression_gates(candidate, prod)
        assert result["decisive_metric"] == "pixel_ap"
        assert result["fallback_to_image_ap"] is False

    def test_no_prod_baseline_is_non_blocking(self) -> None:
        """Without any prod metric, the regression gate cannot be evaluated."""
        result = evaluate_quality_regression_gates(
            candidate_metrics=_metrics(),
            prod_metrics={},
        )
        assert result["all_passed"] is True
        assert result["evaluated_metrics"] == []
        assert result["decisive_metric"] is None

    def test_none_values_are_skipped(self) -> None:
        """A metric present but None (mask absent on one side) is skipped."""
        candidate = _metrics()
        candidate[AUPIMO_KEY] = None
        result = evaluate_quality_regression_gates(candidate, _metrics())
        assert AUPIMO_KEY in result["skipped_metrics"]
        assert result["decisive_metric"] == "pixel_ap"


class TestPromotionGatesWithQualityMetrics:
    """The combined gate folds the 4-metric verdict into the overall decision."""

    def test_quality_regression_blocks_overall(
        self, feature_ae_gates_config: dict
    ) -> None:
        result = evaluate_promotion_gates(
            candidate_recall=1.0,
            candidate_ap=0.90,
            candidate_orange_rate=0.08,
            candidate_latency_ms=900.0,
            gates_config=feature_ae_gates_config,
            candidate_quality_metrics=_metrics(aupimo=0.70),  # big drop
            prod_quality_metrics=_metrics(aupimo=0.80),
        )
        assert result["all_passed"] is False
        assert result["gates"]["quality_regression"]["passed"] is False

    def test_quality_regression_passes_overall(
        self, feature_ae_gates_config: dict
    ) -> None:
        result = evaluate_promotion_gates(
            candidate_recall=1.0,
            candidate_ap=0.91,
            candidate_orange_rate=0.08,
            candidate_latency_ms=900.0,
            gates_config=feature_ae_gates_config,
            candidate_quality_metrics=_metrics(),
            prod_quality_metrics=_metrics(),
        )
        assert result["all_passed"] is True
        assert result["gates"]["quality_regression"]["passed"] is True
        # legacy single-metric ap_regression gate is not added in the 4-metric path
        assert "ap_regression" not in result["gates"]
