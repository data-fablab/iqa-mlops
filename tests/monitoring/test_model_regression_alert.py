"""Contract tests for the IqaModelRegression alert wiring (Issue 5).

Covers the rule expression, threshold coherence with the Issue 4 promotion gate
(no silent duplication), the Alertmanager + webhook-catcher route reuse, and
`promtool test rules` when available.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(".")
RULES = ROOT / "deploy" / "prometheus" / "rules" / "iqa_model_regression.rules.yml"
RULE_TESTS = ROOT / "deploy" / "prometheus" / "rules" / "iqa_model_regression.test.yml"
PROMETHEUS_CFG = ROOT / "deploy" / "prometheus" / "prometheus.yml"
ALERTMANAGER_CFG = ROOT / "deploy" / "alertmanager" / "alertmanager.yml"
GATES_YAML = ROOT / "configs" / "promotion_gates.yaml"


def _alert_rule() -> dict:
    groups = yaml.safe_load(RULES.read_text(encoding="utf-8"))["groups"]
    rules = [r for g in groups for r in g["rules"] if r.get("alert") == "IqaModelRegression"]
    assert len(rules) == 1, "exactly one IqaModelRegression alert expected"
    return rules[0]


class TestAlertRule:
    def test_for_and_labels(self) -> None:
        rule = _alert_rule()
        assert rule["for"] == "1m"
        assert rule["labels"]["severity"] == "critical"
        assert rule["labels"]["signal"] == "model_regression"

    def test_compares_prod_vs_previous_prod_with_image_ap_fallback(self) -> None:
        expr = _alert_rule()["expr"]
        assert 'iqa_model_pixel_aupimo{stage="previous_prod"}' in expr
        assert 'iqa_model_pixel_aupimo{stage="prod"}' in expr
        # Fallback to image_ap, gated on the pixel gauge being absent.
        assert "iqa_model_image_ap" in expr
        assert "absent(iqa_model_pixel_aupimo" in expr

    def test_threshold_is_coherent_with_issue4_gate(self) -> None:
        # No silent duplication: the rule's 0.02 must match the Issue 4 gate's
        # per-metric max regression for pixel_aupimo and image_ap.
        gates = yaml.safe_load(GATES_YAML.read_text(encoding="utf-8"))
        quality = gates["feature_ae"]["quality_max_regression"]
        expr = _alert_rule()["expr"]
        assert f"> {quality['pixel_aupimo_1e-5_1e-3']}" in expr
        assert f"> {quality['image_ap']}" in expr


class TestWiringReuse:
    def test_prometheus_globs_rule_files(self) -> None:
        cfg = yaml.safe_load(PROMETHEUS_CFG.read_text(encoding="utf-8"))
        # The rule file ends with .rules.yml and is picked up by the glob.
        assert any("rules" in path for path in cfg["rule_files"])
        assert RULES.name.endswith(".rules.yml")

    def test_alertmanager_routes_all_alerts_to_webhook_catcher(self) -> None:
        cfg = yaml.safe_load(ALERTMANAGER_CFG.read_text(encoding="utf-8"))
        # A single catch-all route already forwards IqaModelRegression too.
        assert cfg["route"]["receiver"] == "webhook-catcher"


@pytest.mark.skipif(shutil.which("promtool") is None, reason="promtool not installed")
def test_promtool_rule_unit_tests_pass() -> None:
    result = subprocess.run(
        ["promtool", "test", "rules", RULE_TESTS.name],
        cwd=RULE_TESTS.parent, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
