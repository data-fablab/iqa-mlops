from __future__ import annotations

import json
from pathlib import Path


DASHBOARD_DIR = Path("deploy/grafana/provisioning/dashboards/json")


def _dashboard(name: str) -> dict:
    return json.loads((DASHBOARD_DIR / name).read_text(encoding="utf-8"))


def _dashboard_text(dashboard: dict) -> str:
    return json.dumps(dashboard, sort_keys=True)


def test_dashboards_have_at_most_five_panels_each() -> None:
    for name in ("iqa-lifecycle.json", "iqa-drift-p4.json"):
        dashboard = _dashboard(name)
        assert len(dashboard["panels"]) <= 5


def test_lifecycle_dashboard_references_expected_metrics_only() -> None:
    dashboard = _dashboard("iqa-lifecycle.json")
    text = _dashboard_text(dashboard)

    expected_metrics = {
        "iqa_lifecycle_run_events_processed",
        "iqa_lifecycle_run_cycles_completed",
        "iqa_lifecycle_epoch_metric",
        "iqa_lifecycle_gate_value",
        "iqa_lifecycle_promotion_decision_info",
        "iqa_lifecycle_final_model_info",
    }
    for metric in expected_metrics:
        assert metric in text
    assert ".cache" not in text
    assert "windows.jsonl" not in text


def test_drift_dashboard_references_expected_metrics_only() -> None:
    dashboard = _dashboard("iqa-drift-p4.json")
    text = _dashboard_text(dashboard)

    expected_metrics = {
        "iqa_drift_domain_ratio",
        "iqa_drift_score",
        "iqa_drift_status",
        "iqa_drift_first_confirmed_window",
        "iqa_drift_trigger_lifecycle",
        "iqa_drift_window_index",
        "iqa_drift_red_rate",
        "iqa_drift_unexpected_red_rate",
        "iqa_drift_alert_rate",
        "iqa_lifecycle_gate_value",
        "iqa_lifecycle_promotion_decision_info",
        "iqa_lifecycle_final_model_info",
    }
    for metric in expected_metrics:
        assert metric in text
    assert ".cache" not in text
    assert "windows.jsonl" not in text
