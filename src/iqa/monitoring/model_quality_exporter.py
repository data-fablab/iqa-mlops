"""Prometheus exporter for MLflow model-quality metrics (Issue 1 tracer bullet).

Scope: prove the whole transport chain (exporter -> Prometheus -> Grafana) with a
single metric, the low-FPR pixel AUPIMO. The exporter reads the latest value per
``(model_version, stage)`` from the ``iqa-model-quality`` experiment (the runs that
``task_eval`` logs, Issue 0) and exposes them as a Prometheus gauge::

    iqa_model_pixel_aupimo{model_version="...",stage="prod|candidate"} 0.83

Degrades cleanly: if MLflow is unreachable or the metric is missing, ``/metrics``
still serves with ``iqa_model_quality_exporter_up 0`` (or ``1`` with no AUPIMO
samples) instead of crashing, so the Prometheus target stays scrapable.

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
    TAG_MODEL_VERSION,
    TAG_STAGE,
)

# Gauge names (stable contract for Prometheus scrape config + Grafana panels).
AUPIMO_GAUGE = "iqa_model_pixel_aupimo"
EXPORTER_UP_GAUGE = "iqa_model_quality_exporter_up"

DEFAULT_PORT = 9105
# Prometheus text exposition content type (text format 0.0.4).
_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


@dataclass(frozen=True)
class AupimoSample:
    """One gauge point: latest AUPIMO for a (model_version, stage) pair."""

    model_version: str
    stage: str
    value: float


def latest_aupimo_per_model(runs: Iterable[Any]) -> list[AupimoSample]:
    """Reduce MLflow runs to the latest AUPIMO per ``(model_version, stage)``.

    "Latest" is the run with the highest ``info.start_time``. Runs missing the
    AUPIMO metric or either tag are skipped. Output is sorted by
    ``(stage, model_version)`` for stable, diff-friendly exposition.
    """
    latest: dict[tuple[str, str], tuple[int, float]] = {}
    for run in runs:
        metrics = getattr(run.data, "metrics", None) or {}
        tags = getattr(run.data, "tags", None) or {}
        value = metrics.get(AUPIMO_KEY)
        model_version = tags.get(TAG_MODEL_VERSION)
        stage = tags.get(TAG_STAGE)
        if value is None or not model_version or not stage:
            continue
        start_time = int(getattr(run.info, "start_time", 0) or 0)
        key = (model_version, stage)
        current = latest.get(key)
        if current is None or start_time >= current[0]:
            latest[key] = (start_time, float(value))
    return [
        AupimoSample(model_version=model_version, stage=stage, value=value)
        for (model_version, stage), (_, value) in sorted(
            latest.items(), key=lambda item: (item[0][1], item[0][0])
        )
    ]


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def render_metrics(samples: list[AupimoSample], *, exporter_up: bool) -> str:
    """Render the Prometheus text exposition for the AUPIMO gauge + health gauge."""
    lines = [
        f"# HELP {EXPORTER_UP_GAUGE} 1 if the last MLflow scrape succeeded, 0 otherwise",
        f"# TYPE {EXPORTER_UP_GAUGE} gauge",
        f"{EXPORTER_UP_GAUGE} {1 if exporter_up else 0}",
        f"# HELP {AUPIMO_GAUGE} Latest low-FPR pixel AUPIMO per model_version/stage "
        f"(MLflow experiment {MODEL_QUALITY_EXPERIMENT})",
        f"# TYPE {AUPIMO_GAUGE} gauge",
    ]
    for sample in samples:
        lines.append(
            f"{AUPIMO_GAUGE}{{"
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
    return render_metrics(latest_aupimo_per_model(runs), exporter_up=True)


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
