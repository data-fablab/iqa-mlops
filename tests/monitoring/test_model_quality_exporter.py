"""Unit tests for the model-quality Prometheus exporter (Issues 1-2).

The MLflow -> gauge transformation is pure and tested without a live MLflow:
latest run per (model_version, stage), the 4 business gauges, missing metrics
tolerated, and clean degradation when MLflow is unreachable.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from iqa.monitoring import model_quality_exporter as mqe
from iqa.monitoring.model_metrics import (
    AUPIMO_KEY,
    MODEL_QUALITY_METRIC_KEYS,
    TAG_MODEL_VERSION,
    TAG_STAGE,
)

pytestmark = pytest.mark.unit


def _run(*, model_version: str, stage: str, metrics: dict[str, float], start_time: int) -> SimpleNamespace:
    tags = {}
    if model_version:
        tags[TAG_MODEL_VERSION] = model_version
    if stage:
        tags[TAG_STAGE] = stage
    return SimpleNamespace(
        data=SimpleNamespace(metrics=dict(metrics), tags=tags),
        info=SimpleNamespace(start_time=start_time),
    )


_FULL = {AUPIMO_KEY: 0.42, "pixel_ap": 0.61, "image_ap": 0.87, "image_auroc": 0.93}


class TestLatestMetricsPerModel:
    def test_emits_all_four_metrics_for_a_pair(self) -> None:
        samples = mqe.latest_metrics_per_model([_run(model_version="m1", stage="prod", metrics=_FULL, start_time=1)])

        by_key = {s.metric_key: s.value for s in samples}
        assert by_key == _FULL
        assert {(s.model_version, s.stage) for s in samples} == {("m1", "prod")}

    def test_latest_run_wins_per_model_and_stage(self) -> None:
        runs = [
            _run(model_version="m1", stage="prod", metrics={**_FULL, "image_ap": 0.70}, start_time=100),
            _run(model_version="m1", stage="prod", metrics={**_FULL, "image_ap": 0.88}, start_time=200),  # newer
        ]

        samples = mqe.latest_metrics_per_model(runs)

        image_ap = next(s.value for s in samples if s.metric_key == "image_ap")
        assert image_ap == 0.88

    def test_candidate_and_prod_coexist(self) -> None:
        runs = [
            _run(model_version="prod_m", stage="prod", metrics=_FULL, start_time=1),
            _run(model_version="cand_m", stage="candidate", metrics=_FULL, start_time=1),
        ]

        pairs = {(s.model_version, s.stage) for s in mqe.latest_metrics_per_model(runs)}

        assert pairs == {("prod_m", "prod"), ("cand_m", "candidate")}

    def test_tolerates_missing_metrics_and_skips_untagged(self) -> None:
        runs = [
            _run(model_version="m1", stage="prod", metrics={AUPIMO_KEY: 0.42}, start_time=1),  # only aupimo
            _run(model_version="", stage="prod", metrics=_FULL, start_time=1),  # no model_version -> skipped
        ]

        samples = mqe.latest_metrics_per_model(runs)

        assert [(s.metric_key, s.model_version) for s in samples] == [(AUPIMO_KEY, "m1")]


class TestRenderMetrics:
    def test_renders_four_gauges_with_labels(self) -> None:
        samples = mqe.latest_metrics_per_model([_run(model_version="m1", stage="prod", metrics=_FULL, start_time=1)])

        text = mqe.render_metrics(samples, exporter_up=True)

        for metric_key in MODEL_QUALITY_METRIC_KEYS:
            gauge = mqe.GAUGE_NAMES[metric_key]
            assert f"# TYPE {gauge} gauge" in text
            assert f'{gauge}{{model_version="m1",stage="prod"}} ' in text
        assert f"{mqe.EXPORTER_UP_GAUGE} 1" in text

    def test_marks_exporter_down_with_no_samples(self) -> None:
        text = mqe.render_metrics([], exporter_up=False)

        assert f"{mqe.EXPORTER_UP_GAUGE} 0" in text
        # HELP/TYPE for all gauges still present, but no sample lines.
        for metric_key in MODEL_QUALITY_METRIC_KEYS:
            assert f"# TYPE {mqe.GAUGE_NAMES[metric_key]} gauge" in text
        assert "{model_version=" not in text


class TestCollectMetricsText:
    def test_degrades_when_mlflow_unreachable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(**_kwargs: object) -> list[object]:
            raise ConnectionError("mlflow down")

        monkeypatch.setattr(mqe, "_fetch_runs", boom)

        text = mqe.collect_metrics_text(tracking_uri="http://nope:5000")

        assert f"{mqe.EXPORTER_UP_GAUGE} 0" in text

    def test_up_with_no_samples_when_metrics_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runs = [_run(model_version="m1", stage="prod", metrics={}, start_time=1)]
        monkeypatch.setattr(mqe, "_fetch_runs", lambda **_kwargs: runs)

        text = mqe.collect_metrics_text()

        assert f"{mqe.EXPORTER_UP_GAUGE} 1" in text
        assert "{model_version=" not in text
