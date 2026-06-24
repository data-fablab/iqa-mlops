"""Contract tests for the IqaDriftProxy alert wiring (issue 04).

Covers the rule expression, the single-authority invariant (rule threshold ==
calibrated artifact threshold), the Alertmanager + webhook-catcher compose wiring,
a stdlib smoke test of the catcher, and `promtool test rules` when available.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

ROOT = Path(".")
RULES = ROOT / "deploy" / "prometheus" / "rules" / "iqa_drift_proxy.rules.yml"
RULE_TESTS = ROOT / "deploy" / "prometheus" / "rules" / "iqa_drift_proxy.test.yml"
PROMETHEUS_CFG = ROOT / "deploy" / "prometheus" / "prometheus.yml"
ALERTMANAGER_CFG = ROOT / "deploy" / "alertmanager" / "alertmanager.yml"
COMPOSE = ROOT / "deploy" / "docker-compose.yml"
CALIBRATION = ROOT / "configs" / "drift_proxy_calibration.yaml"


def _alert_rule() -> dict:
    groups = yaml.safe_load(RULES.read_text(encoding="utf-8"))["groups"]
    rules = [r for g in groups for r in g["rules"] if r.get("alert") == "IqaDriftProxy"]
    assert len(rules) == 1, "exactly one IqaDriftProxy alert expected"
    return rules[0]


class TestAlertRule:
    def test_expr_filters_drift_regime_and_anomaly_decisions(self) -> None:
        expr = _alert_rule()["expr"]
        assert 'scenario_id=~"drift.*"' in expr
        assert 'decision=~"Orange|Rouge"' in expr
        assert "iqa_prediction_total" in expr
        assert "clamp_min" in expr  # guards the zero-traffic denominator

    def test_for_and_labels(self) -> None:
        rule = _alert_rule()
        assert rule["for"] == "30s"
        assert rule["labels"]["severity"] == "critical"
        assert rule["labels"]["signal"] == "drift_proxy"

    def test_threshold_matches_calibrated_artifact(self) -> None:
        # Single-authority invariant: the rule enforces the value calibrated in 03.
        threshold = yaml.safe_load(CALIBRATION.read_text(encoding="utf-8"))["drift_proxy"]["threshold"]
        assert f"> {threshold}" in _alert_rule()["expr"]


class TestPrometheusWiring:
    def test_prometheus_loads_rules_and_alertmanager(self) -> None:
        cfg = yaml.safe_load(PROMETHEUS_CFG.read_text(encoding="utf-8"))
        assert any("rules" in path for path in cfg["rule_files"])
        targets = cfg["alerting"]["alertmanagers"][0]["static_configs"][0]["targets"]
        assert "alertmanager:9093" in targets

    def test_alertmanager_routes_to_webhook_catcher(self) -> None:
        cfg = yaml.safe_load(ALERTMANAGER_CFG.read_text(encoding="utf-8"))
        assert cfg["route"]["receiver"] == "webhook-catcher"
        url = cfg["receivers"][0]["webhook_configs"][0]["url"]
        assert "webhook-catcher:8080" in url


class TestComposeWiring:
    def test_services_present_and_mounted(self) -> None:
        services = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))["services"]
        assert "alertmanager" in services
        assert "webhook-catcher" in services
        prom_volumes = " ".join(services["prometheus"]["volumes"])
        assert "/etc/prometheus/rules" in prom_volumes
        assert "alertmanager" in services["prometheus"]["depends_on"]
        assert services["alertmanager"]["depends_on"] == ["webhook-catcher"]


class TestWebhookCatcher:
    def test_catcher_records_posted_alert(self) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "catcher", ROOT / "deploy" / "webhook-catcher" / "catcher.py"
        )
        catcher = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(catcher)

        server = ThreadingHTTPServer(("127.0.0.1", 0), catcher.Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            payload = json.dumps(
                {"alerts": [{"status": "firing", "labels": {"alertname": "IqaDriftProxy", "severity": "critical"}, "annotations": {"summary": "drift"}}]}
            ).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/alert", data=payload, method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
                assert json.loads(resp.read())["status"] == "received"
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as resp:  # noqa: S310
                last = json.loads(resp.read())["last_alert"]
            assert last["payload"]["alerts"][0]["labels"]["alertname"] == "IqaDriftProxy"
        finally:
            server.shutdown()


@pytest.mark.skipif(shutil.which("promtool") is None, reason="promtool not installed")
def test_promtool_rule_unit_tests_pass() -> None:
    result = subprocess.run(
        ["promtool", "test", "rules", RULE_TESTS.name],
        cwd=RULE_TESTS.parent, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
