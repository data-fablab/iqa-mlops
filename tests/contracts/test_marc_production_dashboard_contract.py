from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path("deploy/streamlit").resolve()))

from marc_lifecycle import aggregate_lots, classification_quality_rows, lifecycle_rows, production_alerts


PRODUCTION_PAGE = Path("deploy/streamlit/pages/1_Dashboard_Responsable_Production.py")
LINEAGE_PAGE = Path("deploy/streamlit/pages/3_Data_Lineage.py")


def test_production_dashboard_exposes_operational_lot_view() -> None:
    page = PRODUCTION_PAGE.read_text(encoding="utf-8")

    for expected in [
        "IQA_MARC_REPLAY_RUN_DIR",
        "events.jsonl",
        "Dossier de run production",
        "Tableau De Bord Production",
        "Situation Des Lots",
        "Decisions Par Lot",
        "Lecture Qualite",
        "Actions Prioritaires",
        "Lots liberables",
        "Lots a isoler",
        "Lots en attente",
        "controle qualite",
    ]:
        assert expected in page

    for technical in ["Registry", "MLflow", "Gate classification", "Lifecycle Feature-AE"]:
        assert technical not in page


def test_production_and_lineage_responsibilities_are_separated() -> None:
    production = PRODUCTION_PAGE.read_text(encoding="utf-8")
    lineage = LINEAGE_PAGE.read_text(encoding="utf-8")
    runbook = Path("docs/runbook-phase1-iqa.md").read_text(encoding="utf-8")

    for expected in [
        "Liberer",
        "Isoler",
        "En attente",
        "scale=alt.Scale",
        "Etat controle qualite",
    ]:
        assert expected in production

    for expected in [
        "Data Lineage",
        "Tracabilite technique",
        "Modeles actifs et finaux",
        "Cycles, gates et promotions",
        "Actif avant",
        "Delta",
        "Registry",
        "MLflow",
        "pixel_aupimo_1e-5_1e-3",
        "pixel_ap",
        "Gate localisation",
        "Gate classification",
        "Recall candidat",
    ]:
        assert expected in lineage

    combined = production + "\n" + lineage + "\n" + runbook
    for expected in [
        "MLflow",
        "MinIO",
        "DVC",
        "IQA_MARC_REPLAY_RUN_DIR",
        "uv run --extra cpu --with streamlit --with requests",
    ]:
        assert expected in combined


def test_marc_lot_aggregation_counts_conformity_and_alerts() -> None:
    events = [
        {"lot_id": "LOT-001", "oracle_verdict": "conforme", "decision": "green", "roi_quality_status": "ok"},
        {"lot_id": "LOT-001", "oracle_verdict": "defective", "decision": "red", "roi_quality_status": "ok"},
        {"lot_id": "LOT-002", "oracle_verdict": "conforme", "decision": "orange", "roi_quality_status": "low"},
    ]

    lots = aggregate_lots(events, active_model="rd_feature_ae_gated_natural_cycle_003")

    assert lots[0]["pieces"] == 2
    assert lots[0]["conformes_gt"] == 1
    assert lots[0]["defauts_gt"] == 1
    assert lots[0]["rouge"] == 1
    assert lots[0]["statut_lot"] == "Non conforme"
    assert lots[1]["orange"] == 1
    assert "roi_fail_rate" not in lots[1]
    assert "LOT-001 contient 1 defaut(s) confirmes." in production_alerts(lots, [])
    assert not any("ROI fail" in alert for alert in production_alerts(lots, []))


def test_marc_classification_quality_compares_model_to_oracle() -> None:
    events = [
        {"lot_id": "LOT-001", "active_model_version": "m1", "oracle_verdict": "conforme", "decision": "green"},
        {"lot_id": "LOT-001", "active_model_version": "m1", "oracle_verdict": "defective", "decision": "green"},
        {"lot_id": "LOT-002", "active_model_version": "m2", "oracle_verdict": "defective", "decision": "red"},
        {"lot_id": "LOT-002", "active_model_version": "m2", "oracle_verdict": "conforme", "decision": "orange"},
    ]

    rows = classification_quality_rows(events, group_key="active_model_version")

    assert rows[0]["active_model_version"] == "m1"
    assert rows[0]["true_good"] == 1
    assert rows[0]["false_negative"] == 1
    assert rows[0]["defect_recall"] == 0.0
    assert rows[1]["active_model_version"] == "m2"
    assert rows[1]["defect_detected"] == 1
    assert rows[1]["false_positive"] == 1
    assert rows[1]["false_positive_orange"] == 1
    assert rows[1]["defect_recall"] == 1.0
    assert rows[1]["alert_precision"] == 0.5


def test_marc_lifecycle_rows_surface_aupimo_and_promotion() -> None:
    rows = lifecycle_rows(
        [
            {
                "cycle_id": "cycle_003",
                "active_model_before": "rd_feature_ae_gated_natural_cycle_002",
                "candidate_version": "rd_feature_ae_gated_natural_cycle_003",
                "evaluation_seen_events": 180,
                "seen_defective": 5,
                "selected_metric": "pixel_aupimo_1e-5_1e-3",
                "selected_metric_value": 0.0059,
                "active_metric_value": 0.004,
                "candidate_metric_value": 0.0059,
                "metric_delta": 0.0019,
                "active_false_negatives": 1,
                "candidate_false_negatives": 0,
                "localization_gate": {"passed": True},
                "classification_gate": {
                    "passed": True,
                    "active_image_recall": 0.8,
                    "candidate_image_recall": 1.0,
                    "image_recall_delta": 0.2,
                },
                "classification_progress": {
                    "improved": True,
                    "non_regression": True,
                    "summary": "improved: FN 1 -> 0",
                },
                "activated_for_next_events": True,
                "activation_scope": "mlflow_registry",
                "metrics": {"pixel_aupimo_1e-5_1e-3": 0.0059, "pixel_ap": 0.004},
                "gate_decision": "passed",
                "promotion_status": "promoted",
                "registry_stage": "test",
                "registry_status": "registered",
                "mlflow_run_id": "abc123",
            }
        ]
    )

    assert rows[0]["pixel_aupimo_1e-5_1e-3"] == 0.0059
    assert rows[0]["pixel_ap"] == 0.004
    assert rows[0]["actif_avant"] == "rd_feature_ae_gated_natural_cycle_002"
    assert rows[0]["active_metric_value"] == 0.004
    assert rows[0]["candidate_metric_value"] == 0.0059
    assert rows[0]["metric_delta"] == 0.0019
    assert rows[0]["active_false_negatives"] == 1
    assert rows[0]["candidate_false_negatives"] == 0
    assert rows[0]["localization_gate"] is True
    assert rows[0]["classification_gate"] is True
    assert rows[0]["classification_progress_improved"] is True
    assert rows[0]["classification_progress_summary"] == "improved: FN 1 -> 0"
    assert rows[0]["active_image_recall"] == 0.8
    assert rows[0]["candidate_image_recall"] == 1.0
    assert rows[0]["image_recall_delta"] == 0.2
    assert rows[0]["activated_for_next_events"] is True
    assert rows[0]["activation_scope"] == "mlflow_registry"
    assert rows[0]["registry"] == "registered"
    assert rows[0]["promotion"] == "promoted"
    assert rows[0]["mlflow_run_id"] == "abc123"
