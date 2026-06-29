"""Prometheus exposition contract for the /metrics endpoints."""

from __future__ import annotations

import pytest
from fastapi.responses import PlainTextResponse
from prometheus_client.parser import text_string_to_metric_families

from iqa.api.main import app as api_app
from iqa.api.main import metrics as api_metrics
from iqa.inference.service import app as inference_app
from iqa.inference.service import metrics as inference_metrics


def _metrics_route(app):
    for route in app.router.routes:
        if getattr(route, "path", None) == "/metrics":
            return route
    raise AssertionError("no /metrics route registered")


@pytest.mark.parametrize(
    ("app", "render"),
    [
        pytest.param(api_app, api_metrics, id="iqa-api"),
        pytest.param(inference_app, inference_metrics, id="iqa-inference"),
    ],
)
def test_metrics_route_serves_prometheus_plain_text(app, render) -> None:
    route = _metrics_route(app)

    assert route.response_class is PlainTextResponse

    body = render()
    assert not body.lstrip().startswith('"')
    assert list(text_string_to_metric_families(body))
