"""Prometheus exporter for MLflow model-quality metrics (Issues 1-2).

Exposes the 4 canonical Feature-AE business metrics per ``(model_version, stage)``
read from the ``iqa-model-quality`` experiment (the runs ``task_eval`` logs,
Issue 0), as Prometheus gauges, so candidate-vs-prod quality is visible end to end
(exporter -> Prometheus -> Grafana)::

    iqa_model_pixel_aupimo{model_version="...",stage="prod|candidate"} 0.83
    iqa_model_pixel_ap{...} ...
    iqa_model_image_ap{...} ...
    iqa_model_image_auroc{...} ...

Each ``(model_version, stage)`` pair contributes its latest run's metrics (highest
``start_time``); missing metrics are tolerated (e.g. pixel metrics when GT masks are
absent). Degrades cleanly: if MLflow is unreachable the endpoint still serves with
``iqa_model_quality_exporter_up 0`` instead of crashing, so the Prometheus target
stays scrapable.

Runnable rebuild-free against mounted source with
``python -m iqa.monitoring.model_quality_exporter``.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterable

from iqa.monitoring.model_metrics import (
    AUPIMO_KEY,
    MODEL_QUALITY_EXPERIMENT,
    MODEL_QUALITY_METRIC_KEYS,
    TAG_MODEL_VERSION,
    TAG_STAGE,
)

# Stable mapping metric key -> Prometheus gauge name (the AUPIMO key carries the
# 1e-5/1e-3 FPR band suffix, which we drop in the gauge name for readability).
# This is the scrape/dashboard contract; keep names stable.
GAUGE_NAMES: dict[str, str] = {
    AUPIMO_KEY: "iqa_model_pixel_aupimo",
    "pixel_ap": "iqa_model_pixel_ap",
    "image_ap": "iqa_model_image_ap",
    "image_auroc": "iqa_model_image_auroc",
}
GAUGE_HELP: dict[str, str] = {
    AUPIMO_KEY: "Latest low-FPR pixel AUPIMO per model_version/stage",
    "pixel_ap": "Latest pixel average precision per model_version/stage",
    "image_ap": "Latest image average precision per model_version/stage",
    "image_auroc": "Latest image AUROC per model_version/stage",
}
# Back-compat alias for the Issue 1 single-metric tracer bullet.
AUPIMO_GAUGE = GAUGE_NAMES[AUPIMO_KEY]
EXPORTER_UP_GAUGE = "iqa_model_quality_exporter_up"

DEFAULT_PORT = 9105
# Prometheus text exposition content type (text format 0.0.4).
_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


@dataclass(frozen=True)
class MetricSample:
    """One gauge point: a business metric value for a (model_version, stage) pair."""

    metric_key: str
    model_version: str
    stage: str
    value: float


def latest_metrics_per_model(runs: Iterable[Any]) -> list[MetricSample]:
    """Reduce MLflow runs to the latest business metrics per ``(model_version, stage)``.

    For each pair, the run with the highest ``info.start_time`` wins; that run's
    present business metrics (out of :data:`MODEL_QUALITY_METRIC_KEYS`) become
    samples. Runs missing either tag are skipped; missing metric values are
    tolerated. Output is sorted by (metric order, stage, model_version) for stable,
    diff-friendly exposition.
    """
    latest_run: dict[tuple[str, str], tuple[int, dict[str, Any]]] = {}
    for run in runs:
        metrics = getattr(run.data, "metrics", None) or {}
        tags = getattr(run.data, "tags", None) or {}
        model_version = tags.get(TAG_MODEL_VERSION)
        stage = tags.get(TAG_STAGE)
        if not model_version or not stage:
            continue
        start_time = int(getattr(run.info, "start_time", 0) or 0)
        key = (model_version, stage)
        current = latest_run.get(key)
        if current is None or start_time >= current[0]:
            latest_run[key] = (start_time, metrics)

    samples: list[MetricSample] = []
    for (model_version, stage), (_, metrics) in latest_run.items():
        for metric_key in MODEL_QUALITY_METRIC_KEYS:
            value = metrics.get(metric_key)
            if value is None:
                continue
            samples.append(
                MetricSample(
                    metric_key=metric_key,
                    model_version=model_version,
                    stage=stage,
                    value=float(value),
                )
            )
    order = {key: index for index, key in enumerate(MODEL_QUALITY_METRIC_KEYS)}
    samples.sort(key=lambda sample: (order[sample.metric_key], sample.stage, sample.model_version))
    return samples


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def render_metrics(samples: list[MetricSample], *, exporter_up: bool) -> str:
    """Render the Prometheus text exposition for the 4 gauges + health gauge.

    HELP/TYPE for every gauge is always emitted (even with no samples) so the
    series are discoverable; sample lines follow per gauge.
    """
    by_metric: dict[str, list[MetricSample]] = {}
    for sample in samples:
        by_metric.setdefault(sample.metric_key, []).append(sample)

    lines = [
        f"# HELP {EXPORTER_UP_GAUGE} 1 if the last MLflow scrape succeeded, 0 otherwise",
        f"# TYPE {EXPORTER_UP_GAUGE} gauge",
        f"{EXPORTER_UP_GAUGE} {1 if exporter_up else 0}",
    ]
    for metric_key in MODEL_QUALITY_METRIC_KEYS:
        gauge = GAUGE_NAMES[metric_key]
        lines.append(
            f"# HELP {gauge} {GAUGE_HELP[metric_key]} (MLflow experiment {MODEL_QUALITY_EXPERIMENT})"
        )
        lines.append(f"# TYPE {gauge} gauge")
        for sample in by_metric.get(metric_key, []):
            lines.append(
                f"{gauge}{{"
                f'model_version="{_escape_label(sample.model_version)}",'
                f'stage="{_escape_label(sample.stage)}"'
                f"}} {sample.value}"
            )
    return "\n".join(lines) + "\n"


def collect_metrics_text(
    *, tracking_uri: str | None = None, experiment: str = MODEL_QUALITY_EXPERIMENT
) -> str:
    """Query MLflow and render the gauges; degrade to ``exporter_up 0`` on failure."""
    try:
        runs = _fetch_runs(tracking_uri=tracking_uri, experiment=experiment)
    except Exception:
        return render_metrics([], exporter_up=False)
    return render_metrics(latest_metrics_per_model(runs), exporter_up=True)


def _fetch_runs(*, tracking_uri: str | None, experiment: str) -> list[Any]:
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    found = client.get_experiment_by_name(experiment)
    if found is None:
        return []
    return list(client.search_runs([found.experiment_id], max_results=1000))


def _make_handler(tracking_uri: str | None) -> type[BaseHTTPRequestHandler]:
    class _MetricsHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            if self.path.rstrip("/") not in ("", "/metrics"):
                self.send_response(404)
                self.end_headers()
                return
            body = collect_metrics_text(tracking_uri=tracking_uri).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", _CONTENT_TYPE)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args: Any) -> None:  # silence per-request stderr noise
            return

    return _MetricsHandler


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI"),
        help="MLflow tracking URI (defaults to MLFLOW_TRACKING_URI).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), _make_handler(args.tracking_uri))
    print(
        f"iqa-model-quality-exporter listening on {args.host}:{args.port} "
        f"(mlflow={args.tracking_uri})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
