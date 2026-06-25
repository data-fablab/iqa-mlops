"""Unit tests for the model-quality Prometheus exporter (Issue 1 tracer bullet).

The MLflow -> gauge transformation is pure and tested without a live MLflow:
latest value per (model_version, stage), and clean degradation when MLflow is
unreachable or the AUPIMO metric is absent.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from iqa.monitoring import model_quality_exporter as mqe
from iqa.monitoring.model_metrics import AUPIMO_KEY, TAG_MODEL_VERSION, TAG_STAGE

pytestmark = pytest.mark.unit


def _run(*, model_version: str, stage: str, aupimo: float | None, start_time: int) -> SimpleNamespace:
    metrics = {} if aupimo is None else {AUPIMO_KEY: aupimo}
    tags = {}
    if model_version:
        tags[TAG_MODEL_VERSION] = model_version
    if stage:
        tags[TAG_STAGE] = stage
    return SimpleNamespace(
        data=SimpleNamespace(metrics=metrics, tags=tags),
        info=SimpleNamespace(start_time=start_time),
    )


class TestLatestAupimoPerModel:
    def test_keeps_latest_value_per_model_and_stage(self) -> None:
        runs = [
            _run(model_version="m1", stage="prod", aupimo=0.70, start_time=100),
            _run(model_version="m1", stage="prod", aupimo=0.83, start_time=200),  # newer wins
            _run(model_version="m1", stage="candidate", aupimo=0.60, start_time=150),
        ]

        samples = mqe.latest_aupimo_per_model(runs)

        by_key = {(s.model_version, s.stage): s.value for s in samples}
        assert by_key == {("m1", "prod"): 0.83, ("m1", "candidate"): 0.60}

    def test_skips_runs_missing_metric_or_tags(self) -> None:
        runs = [
            _run(model_version="m1", stage="prod", aupimo=None, start_time=100),  # no metric
            _run(model_version="", stage="prod", aupimo=0.5, start_time=100),  # no model_version
            _run(model_version="m2", stage="", aupimo=0.5, start_time=100),  # no stage
            _run(model_version="m3", stage="candidate", aupimo=0.42, start_time=100),  # valid
        ]

        samples = mqe.latest_aupimo_per_model(runs)

        assert [(s.model_version, s.stage, s.value) for s in samples] == [("m3", "candidate", 0.42)]


class TestRenderMetrics:
    def test_renders_gauge_with_labels_and_exporter_up(self) -> None:
        samples = [mqe.AupimoSample(model_version="m1", stage="prod", value=0.83)]

        text = mqe.render_metrics(samples, exporter_up=True)

        assert f"# TYPE {mqe.AUPIMO_GAUGE} gauge" in text
        assert f'{mqe.AUPIMO_GAUGE}{{model_version="m1",stage="prod"}} 0.83' in text
        assert f"{mqe.EXPORTER_UP_GAUGE} 1" in text

    def test_marks_exporter_down_with_no_samples(self) -> None:
        text = mqe.render_metrics([], exporter_up=False)

        assert f"{mqe.EXPORTER_UP_GAUGE} 0" in text
        assert mqe.AUPIMO_GAUGE in text  # HELP/TYPE still present
        assert "{model_version=" not in text  # but no sample lines


class TestCollectMetricsText:
    def test_degrades_when_mlflow_unreachable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(**_kwargs: object) -> list[object]:
            raise ConnectionError("mlflow down")

        monkeypatch.setattr(mqe, "_fetch_runs", boom)

        text = mqe.collect_metrics_text(tracking_uri="http://nope:5000")

        assert f"{mqe.EXPORTER_UP_GAUGE} 0" in text

    def test_up_with_no_samples_when_metric_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runs = [_run(model_version="m1", stage="prod", aupimo=None, start_time=1)]
        monkeypatch.setattr(mqe, "_fetch_runs", lambda **_kwargs: runs)

        text = mqe.collect_metrics_text()

        assert f"{mqe.EXPORTER_UP_GAUGE} 1" in text
        assert "{model_version=" not in text
