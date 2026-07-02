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
        "iqa_drift_roi_mask_novelty_rate",
        "iqa_drift_roi_mask_nn_distance",
        "iqa_drift_context_events_total",
        "iqa_drift_status",
        "iqa_drift_trigger_lifecycle",
        "iqa_lifecycle_run_events_processed",
        "iqa_lifecycle_promotion_total",
        "iqa_lifecycle_promotion_selected_epoch",
        "iqa_lifecycle_epoch_current",
        "iqa_lifecycle_epoch_image_ap",
        "iqa_lifecycle_epoch_pixel_ap",
        "iqa_lifecycle_epoch_pixel_aupimo",
        "iqa_lifecycle_phase_active",
        "iqa_lifecycle_gate_value",
        "iqa_lifecycle_final_model_info",
    }
    for metric in expected_metrics:
        assert metric in text


def test_drift_dashboard_exposes_story_arc_and_correction_sections() -> None:
    dashboard = _dashboard("iqa-drift-p4.json")
    panels_by_title = {panel["title"]: panel for panel in dashboard["panels"]}

    expected_titles = {
        "Drift P4 puis entrainement correctif",
        "Chronologie scenario - drift puis correction",
        "Preuve ROI - Piece B stable puis P4",
        "Correction ciblee - contexte et train set",
        "Phase DAG correctif - entrainement gate promotion",
        "Epochs correction - progression",
        "Metriques entrainement correctif - progression",
        "Gate classification - avant apres",
        "Gate localisation - avant apres",
        "Classification promue",
        "Localisation promue",
        "Epoch classification",
        "Epoch localisation",
        "Registry correction - modeles promus",
    }
    assert expected_titles <= set(panels_by_title)

    assert "Preuve Piece B report-only" not in panels_by_title
    assert "Fenetres decisionnelles du drift P4" not in panels_by_title
    assert "Preuve drift - signaux terrain" not in panels_by_title
    assert "Preuve drift - signaux critiques" not in panels_by_title
    assert "Live drift - P4 detecte puis correction lancee" not in panels_by_title
    assert "Preuve drift - signatures ROI" not in panels_by_title
    assert "Entrainement correctif - contexte cible" not in panels_by_title
    assert "Entrainement correctif - progression" not in panels_by_title
    assert "Live entrainement - epoch, train set et evenements" not in panels_by_title
    assert "Promotion checkpoints - epoch choisie et roles promus" not in panels_by_title
    assert "Phase DAG correctif - entrainement evaluation gate promotion" not in panels_by_title
    assert "Entrainement puis evaluation - progression" not in panels_by_title
    assert "Entrainement puis gate - progression" not in panels_by_title
    assert "Piece B stable - novelty ROI" not in panels_by_title
    assert "P4 hors referentiel - novelty ROI" not in panels_by_title
    assert "Fenetre observee" not in panels_by_title
    assert "Drift confirme" not in panels_by_title
    assert "DAG correctif" not in panels_by_title
    assert "Train set correctif" not in panels_by_title
    assert "Promotions" not in panels_by_title
    assert "Epoch correctif" not in panels_by_title
    assert "Decision finale" not in panels_by_title

    timeseries_titles = {
        panel["title"] for panel in dashboard["panels"] if panel["type"] == "timeseries"
    }
    assert {
        "Chronologie scenario - drift puis correction",
        "Preuve ROI - Piece B stable puis P4",
        "Correction ciblee - contexte et train set",
        "Epochs correction - progression",
        "Metriques entrainement correctif - progression",
    } <= timeseries_titles

    chronology = panels_by_title["Chronologie scenario - drift puis correction"]
    chronology_expr = " ".join(target["expr"] for target in chronology["targets"])
    assert chronology["type"] == "timeseries"
    assert chronology["gridPos"]["w"] == 24
    assert "iqa_drift_roi_mask_novelty_rate" not in chronology_expr
    assert "vector(0.8)" not in chronology_expr
    assert "iqa_drift_status" in chronology_expr
    assert "iqa_drift_trigger_lifecycle" in chronology_expr
    assert "iqa_lifecycle_promotion_total" in chronology_expr

    roi = panels_by_title["Preuve ROI - Piece B stable puis P4"]
    roi_expr = " ".join(target["expr"] for target in roi["targets"])
    assert roi["type"] == "timeseries"
    assert roi["gridPos"]["w"] == 24
    assert "iqa_drift_roi_mask_novelty_rate" in roi_expr
    assert "iqa_drift_roi_mask_nn_distance" in roi_expr
    assert "vector(0.8)" in roi_expr
    assert "iqa_drift_domain_ratio" not in roi_expr
    assert "iqa_drift_roi_fail_rate" not in roi_expr
    assert "iqa_drift_oracle_fn_rate" not in roi_expr

    correction = panels_by_title["Correction ciblee - contexte et train set"]
    correction_expr = " ".join(target["expr"] for target in correction["targets"])
    assert correction["type"] == "timeseries"
    assert correction["gridPos"]["w"] == 24
    assert "iqa_drift_context_events_total" in correction_expr
    assert "iqa_drift_window_events" in correction_expr
    assert "iqa_lifecycle_train_set_size" not in correction_expr
    assert "iqa_lifecycle_run_events_processed" in correction_expr

    phase_timeline = panels_by_title["Phase DAG correctif - entrainement gate promotion"]
    phase_expr = " ".join(target["expr"] for target in phase_timeline["targets"])
    phase_legends = [target["legendFormat"] for target in phase_timeline["targets"]]
    assert phase_timeline["type"] == "timeseries"
    assert phase_timeline["gridPos"]["w"] == 24
    assert "iqa_lifecycle_phase_active" in phase_expr
    assert 'phase="training"' in phase_expr
    assert 'phase="evaluation_gate"' in phase_expr
    assert 'phase="promotion"' in phase_expr
    assert "iqa_lifecycle_epoch_current" not in phase_expr
    assert "iqa_lifecycle_gate_value" not in phase_expr
    assert "iqa_lifecycle_promotion_selected_epoch" not in phase_expr
    assert "max_over_time" not in phase_expr
    assert "@ end()" not in phase_expr
    assert phase_legends == ["Entrainement", "Gate", "Promotion"]

    epochs = panels_by_title["Epochs correction - progression"]
    epoch_expr = " ".join(target["expr"] for target in epochs["targets"])
    assert epochs["type"] == "timeseries"
    assert epochs["gridPos"]["w"] == 24
    assert epoch_expr == 'iqa_lifecycle_epoch_current{scenario_id="production_replay_natural_piece_b_to_piece_a_p4_drift"}'
    assert [target["legendFormat"] for target in epochs["targets"]] == ["epoch"]

    training_metrics = panels_by_title["Metriques entrainement correctif - progression"]
    training_expr = " ".join(target["expr"] for target in training_metrics["targets"])
    training_legends = [target["legendFormat"] for target in training_metrics["targets"]]
    assert training_metrics["type"] == "timeseries"
    assert training_metrics["gridPos"]["w"] == 24
    assert "iqa_lifecycle_epoch_image_ap" in training_expr
    assert "iqa_lifecycle_epoch_pixel_ap" in training_expr
    assert "iqa_lifecycle_epoch_pixel_aupimo" in training_expr
    assert "iqa_lifecycle_epoch_current" not in training_expr
    assert "iqa_lifecycle_phase_active" not in training_expr
    assert "iqa_lifecycle_gate_value" not in training_expr
    assert "iqa_lifecycle_promotion_selected_epoch" not in training_expr
    assert "max_over_time" not in training_expr
    assert "@ end()" not in training_expr
    assert training_legends == [
        "{{role}} image AP",
        "{{role}} pixel AP",
        "{{role}} pixel AUPIMO",
    ]

    classification_gate = panels_by_title["Gate classification - avant apres"]
    classification_expr = " ".join(target["expr"] for target in classification_gate["targets"])
    classification_legends = [target["legendFormat"] for target in classification_gate["targets"]]
    assert classification_gate["type"] == "bargauge"
    assert "iqa_lifecycle_gate_value" in classification_expr
    assert 'role="classification"' in classification_expr
    assert 'model="active"' in classification_expr
    assert 'model="candidate"' in classification_expr
    assert "false_negatives" in classification_expr
    assert "image_recall" in classification_expr
    assert "image_ap" in classification_expr
    assert classification_legends == [
        "FN active",
        "FN candidat",
        "Image AP active",
        "Image AP candidat",
        "Recall active",
        "Recall candidat",
    ]

    localization_gate = panels_by_title["Gate localisation - avant apres"]
    localization_expr = " ".join(target["expr"] for target in localization_gate["targets"])
    localization_legends = [target["legendFormat"] for target in localization_gate["targets"]]
    assert localization_gate["type"] == "bargauge"
    assert "iqa_lifecycle_gate_value" in localization_expr
    assert 'role="localization"' in localization_expr
    assert 'model="active"' in localization_expr
    assert 'model="candidate"' in localization_expr
    assert "false_negatives" in localization_expr
    assert "pixel_ap" in localization_expr
    assert "pixel_aupimo" in localization_expr
    assert localization_legends == [
        "FN active",
        "FN candidat",
        "Pixel AP active",
        "Pixel AP candidat",
        "AUPIMO active",
        "AUPIMO candidat",
    ]

    for title, role in (
        ("Epoch classification", "classification"),
        ("Epoch localisation", "localization"),
    ):
        expr = panels_by_title[title]["targets"][0]["expr"]
        assert "iqa_lifecycle_promotion_selected_epoch" in expr
        assert f'role="{role}"' in expr

    correction_table = panels_by_title["Registry correction - modeles promus"]
    assert correction_table["gridPos"]["w"] == 24
    assert "iqa_lifecycle_final_model_info" in correction_table["targets"][0]["expr"]

    custom_annotations = {
        annotation.get("name"): annotation
        for annotation in dashboard["annotations"]["list"]
        if not annotation.get("builtIn")
    }
    assert custom_annotations == {}


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
