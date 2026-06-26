"""Contract tests for PatchCore domain-drift metrics (Issue 14).

Verifies:
- ``_record_prediction_metrics`` increments the regime counter and updates the
  drift-score gauge when a prediction carries ``domain_regime`` / ``domain_drift_score``.
- ``/metrics`` renders ``iqa_domain_drift_total`` (counter by regime) and
  ``iqa_domain_drift_score`` (gauge) in Prometheus text exposition.
"""

from __future__ import annotations

import pytest

from iqa.api.main import (
    DOMAIN_DRIFT_METRICS,
    DOMAIN_DRIFT_REGIME_COUNTS,
    _record_prediction_metrics,
    metrics as api_metrics,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_domain_drift_state():
    saved_counts = dict(DOMAIN_DRIFT_REGIME_COUNTS)
    saved_metrics = dict(DOMAIN_DRIFT_METRICS)
    DOMAIN_DRIFT_REGIME_COUNTS.clear()
    DOMAIN_DRIFT_METRICS["last_score"] = 0.0
    yield
    DOMAIN_DRIFT_REGIME_COUNTS.clear()
    DOMAIN_DRIFT_REGIME_COUNTS.update(saved_counts)
    DOMAIN_DRIFT_METRICS.update(saved_metrics)


def _prediction(*, regime: str | None = None, score: float | None = None) -> dict:
    return {
        "decision": "Vert",
        "roi_status": "ok",
        "domain_regime": regime,
        "domain_drift_score": score,
    }


class TestRecordPredictionMetricsDomainDrift:
    def test_increments_in_domain_counter(self) -> None:
        _record_prediction_metrics(_prediction(regime="in_domain", score=2.5), 0.01, "s1")
        assert DOMAIN_DRIFT_REGIME_COUNTS.get("in_domain") == 1
        assert DOMAIN_DRIFT_REGIME_COUNTS.get("out_of_domain", 0) == 0

    def test_increments_out_of_domain_counter(self) -> None:
        _record_prediction_metrics(_prediction(regime="out_of_domain", score=4.2), 0.01, "s1")
        assert DOMAIN_DRIFT_REGIME_COUNTS.get("out_of_domain") == 1

    def test_updates_drift_score_gauge(self) -> None:
        _record_prediction_metrics(_prediction(regime="in_domain", score=2.78), 0.01, "s1")
        assert DOMAIN_DRIFT_METRICS["last_score"] == pytest.approx(2.78)
        _record_prediction_metrics(_prediction(regime="out_of_domain", score=4.22), 0.01, "s1")
        assert DOMAIN_DRIFT_METRICS["last_score"] == pytest.approx(4.22)

    def test_ignores_none_regime(self) -> None:
        _record_prediction_metrics(_prediction(regime=None, score=None), 0.01, "s1")
        assert DOMAIN_DRIFT_REGIME_COUNTS.get("in_domain", 0) == 0
        assert DOMAIN_DRIFT_REGIME_COUNTS.get("out_of_domain", 0) == 0

    def test_ignores_unknown_regime(self) -> None:
        _record_prediction_metrics(_prediction(regime="unknown", score=1.0), 0.01, "s1")
        assert "unknown" not in DOMAIN_DRIFT_REGIME_COUNTS

    def test_cumulates_counts(self) -> None:
        for _ in range(3):
            _record_prediction_metrics(_prediction(regime="in_domain", score=2.0), 0.01, "s1")
        _record_prediction_metrics(_prediction(regime="out_of_domain", score=5.0), 0.01, "s1")
        assert DOMAIN_DRIFT_REGIME_COUNTS["in_domain"] == 3
        assert DOMAIN_DRIFT_REGIME_COUNTS["out_of_domain"] == 1


class TestMetricsEndpointDomainDrift:
    def test_renders_drift_total_counter_lines(self) -> None:
        DOMAIN_DRIFT_REGIME_COUNTS["in_domain"] = 7
        DOMAIN_DRIFT_REGIME_COUNTS["out_of_domain"] = 3
        body = api_metrics()
        assert 'iqa_domain_drift_total{regime="in_domain"} 7' in body
        assert 'iqa_domain_drift_total{regime="out_of_domain"} 3' in body

    def test_renders_drift_total_zero_when_no_traffic(self) -> None:
        body = api_metrics()
        assert 'iqa_domain_drift_total{regime="in_domain"} 0' in body
        assert 'iqa_domain_drift_total{regime="out_of_domain"} 0' in body

    def test_renders_drift_score_gauge(self) -> None:
        DOMAIN_DRIFT_METRICS["last_score"] = 3.14
        body = api_metrics()
        assert "iqa_domain_drift_score 3.14" in body

    def test_drift_metrics_type_annotations(self) -> None:
        body = api_metrics()
        assert "# TYPE iqa_domain_drift_total counter" in body
        assert "# TYPE iqa_domain_drift_score gauge" in body
