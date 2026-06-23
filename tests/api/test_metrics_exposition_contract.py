"""Prometheus exposition contract for the ``/metrics`` endpoints.

Regression guard for the FastAPI default-serialization bug: a route declared
``def metrics() -> str`` is serialized as JSON (``application/json``, quoted body
with literal ``\\n``), which Prometheus cannot scrape. Both services must expose
``/metrics`` as parseable ``text/plain`` via ``PlainTextResponse``.

Introspects the route's ``response_class`` (no HTTP client / httpx dependency)
and parses the rendered body with ``prometheus_client``.
"""

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
def test_metrics_route_serves_plain_text(app, render) -> None:
    route = _metrics_route(app)

    # Guards against a regression to a bare ``-> str`` route (JSON-serialized).
    assert route.response_class is PlainTextResponse

    body = render()
    # Not a JSON-quoted blob.
    assert not body.lstrip().startswith('"')
    # Parses as Prometheus text exposition (>= 1 metric family).
    families = list(text_string_to_metric_families(body))
    assert families
