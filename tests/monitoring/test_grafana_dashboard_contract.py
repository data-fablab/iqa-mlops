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
        "iqa_lifecycle_cycle_current",
        "iqa_lifecycle_epoch_current",
        "iqa_lifecycle_active_model_info",
        "iqa_lifecycle_epoch_pixel_aupimo",
        "iqa_lifecycle_epoch_pixel_ap",
        "iqa_lifecycle_epoch_image_ap",
        "iqa_lifecycle_promotion_total",
        "iqa_lifecycle_promotion_selected_epoch",
        "iqa_lifecycle_train_set_size",
    }
    for metric in expected_metrics:
        assert metric in text


def test_lifecycle_progression_keeps_pixel_and_image_scales_separate() -> None:
    dashboard = _dashboard("iqa-lifecycle.json")
    panels_by_title = {panel["title"]: panel for panel in dashboard["panels"]}

    expected_panels = {
        "Pixel AUPIMO par epoch": "iqa_lifecycle_epoch_pixel_aupimo",
        "Pixel AP par epoch": "iqa_lifecycle_epoch_pixel_ap",
        "Image AP par epoch": "iqa_lifecycle_epoch_image_ap",
    }
    for title, metric in expected_panels.items():
        panel = panels_by_title[title]
        expressions = [target["expr"] for target in panel["targets"]]
        assert expressions == [metric]
        assert "{{cycle_id}}" in panel["targets"][0]["legendFormat"]


def test_lifecycle_dashboard_shows_train_set_growth_per_cycle() -> None:
    dashboard = _dashboard("iqa-lifecycle.json")
    panels_by_title = {panel["title"]: panel for panel in dashboard["panels"]}

    panel = panels_by_title["Taille train set par cycle"]
    assert panel["gridPos"]["w"] == 24
    expressions = [target["expr"] for target in panel["targets"]]
    assert len(expressions) == 3
    assert all("iqa_lifecycle_train_set_size" in expr for expr in expressions)
    assert any('kind="total"' in expr for expr in expressions)
    assert any('kind="seen_conforming"' in expr for expr in expressions)
    assert any('kind="anchor_good"' in expr for expr in expressions)

    table = panels_by_title["Train sets utilises par cycle"]
    assert table["gridPos"]["w"] == 24
    assert table["targets"][0]["format"] == "table"
    assert 'kind="total"' in table["targets"][0]["expr"]


def test_lifecycle_dashboard_marks_promoted_cycles_with_annotations() -> None:
    dashboard = _dashboard("iqa-lifecycle.json")
    annotations = dashboard["annotations"]["list"]

    promotion_annotations = {
        annotation.get("name"): annotation
        for annotation in annotations
        if str(annotation.get("name", "")).startswith("Decision gate - ")
    }
    assert "Decision gate - localisation" in promotion_annotations
    assert "Decision gate - classification" in promotion_annotations
    for role, annotation_name in (
        ("localization", "Decision gate - localisation"),
        ("classification", "Decision gate - classification"),
    ):
        annotation = promotion_annotations[annotation_name]
        assert annotation["enable"] is True
        assert "iqa_lifecycle_promotion_total" in annotation["expr"]
        assert f'role="{role}"' in annotation["expr"]
        assert 'status="promoted"' in annotation["expr"]
    panel_titles = {panel["title"] for panel in dashboard["panels"]}
    assert "Epoch localisation" in panel_titles
    assert "Epoch classification" in panel_titles
    assert "Decision gate - lecture metier" not in panel_titles
    promotion_table = next(panel for panel in dashboard["panels"] if panel["title"] == "Cycles avec promotions - modele et epoch promue")
    assert promotion_table["gridPos"]["w"] == 24


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
