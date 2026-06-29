from __future__ import annotations

import json
from pathlib import Path


DASHBOARD_DIR = Path("deploy/grafana/provisioning/dashboards/json")
NARRATIVE_DASHBOARDS = {
    "iqa-executive-mlops.json": "iqa-executive-mlops",
    "iqa-lifecycle.json": "iqa-lifecycle",
    "iqa-drift-p4.json": "iqa-drift-p4",
}


def _dashboard(name: str) -> dict:
    return json.loads((DASHBOARD_DIR / name).read_text(encoding="utf-8"))


def _dashboard_text(dashboard: dict) -> str:
    return json.dumps(dashboard, sort_keys=True)


def test_narrative_dashboards_exist_with_stable_uids() -> None:
    for name, uid in NARRATIVE_DASHBOARDS.items():
        dashboard = _dashboard(name)
        assert dashboard["uid"] == uid
        assert dashboard["title"].startswith("IQA - ")
        assert len(dashboard["panels"]) >= 6


def test_narrative_dashboards_do_not_expose_sensitive_artifacts() -> None:
    forbidden_fragments = {
        ".cache",
        "windows.jsonl",
        "validation_gt_masks",
        "gt_masks",
        "mask_uri",
        "file://",
        "C:\\",
        "D:\\",
    }
    for name in NARRATIVE_DASHBOARDS:
        text = _dashboard_text(_dashboard(name))
        for fragment in forbidden_fragments:
            assert fragment not in text


def test_lifecycle_dashboard_references_expected_metrics_only() -> None:
    dashboard = _dashboard("iqa-lifecycle.json")
    text = _dashboard_text(dashboard)

    expected_metrics = {
        "iqa_lifecycle_run_events_processed",
        "iqa_lifecycle_run_cycles_completed",
        "iqa_lifecycle_epoch_metric",
        "iqa_lifecycle_epoch_pixel_aupimo",
        "iqa_lifecycle_epoch_pixel_ap",
        "iqa_lifecycle_epoch_image_ap",
        "iqa_lifecycle_gate_value",
        "iqa_lifecycle_gate_metric_delta",
        "iqa_lifecycle_gate_fn_delta",
        "iqa_lifecycle_promotion_total",
        "iqa_lifecycle_promotion_decision_info",
        "iqa_lifecycle_final_model_info",
    }
    for metric in expected_metrics:
        assert metric in text


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
        "iqa_drift_roi_fail_rate",
        "iqa_drift_oracle_fn_rate",
        "iqa_drift_active_model_info",
        "iqa_lifecycle_gate_value",
        "iqa_lifecycle_promotion_decision_info",
        "iqa_lifecycle_final_model_info",
    }
    for metric in expected_metrics:
        assert metric in text


def test_executive_dashboard_references_expected_metrics_only() -> None:
    dashboard = _dashboard("iqa-executive-mlops.json")
    text = _dashboard_text(dashboard)

    expected_metrics = {
        "iqa_api_up",
        "iqa_inference_up",
        "iqa_lifecycle_run_cycles_completed",
        "iqa_lifecycle_epoch_metric",
        "iqa_lifecycle_epoch_pixel_aupimo",
        "iqa_drift_domain_ratio",
        "iqa_drift_score",
        "iqa_drift_status",
        "iqa_drift_trigger_lifecycle",
        "iqa_lifecycle_promotion_total",
        "iqa_lifecycle_promotion_decision_info",
        "iqa_lifecycle_final_model_info",
    }
    for metric in expected_metrics:
        assert metric in text


def test_narrative_dashboards_expose_storytelling_titles() -> None:
    combined_text = "\n".join(_dashboard_text(_dashboard(name)) for name in NARRATIVE_DASHBOARDS)

    expected_titles = {
        "Fil rouge",
        "Premiere semaine",
        "Drift confirme",
        "Correction",
        "Registry",
    }
    for title in expected_titles:
        assert title in combined_text
