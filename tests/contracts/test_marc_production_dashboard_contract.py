from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path("deploy/streamlit").resolve()))

from marc_lifecycle import aggregate_lots, lifecycle_rows, production_alerts


def test_marc_dashboard_exposes_lifecycle_run_and_api_history() -> None:
    page = Path("deploy/streamlit/pages/1_Dashboard_Marc.py").read_text(encoding="utf-8")

    for expected in [
        "IQA_MARC_REPLAY_RUN_DIR",
        "events.jsonl",
        "lots.jsonl",
        "cycles.jsonl",
        "summary.json",
        "progress.json",
        "lifecycle_events.jsonl",
        "Run lifecycle",
        "Historique API",
        "/lots/summary",
    ]:
        assert expected in page


def test_marc_dashboard_exposes_production_and_lineage_terms() -> None:
    page = Path("deploy/streamlit/pages/1_Dashboard_Marc.py").read_text(encoding="utf-8")
    runbook = Path("docs/runbook-phase1-iqa.md").read_text(encoding="utf-8")
    combined = page + "\n" + runbook

    for expected in [
        "Conformite des lots",
        "Lifecycle Feature-AE",
        "Modele actif courant",
        "Journal lifecycle live",
        "Actif avant",
        "Delta",
        "Registry",
        "pixel_aupimo_1e-5_1e-3",
        "pixel_ap",
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
    assert lots[0]["statut_lot"] == "A revoir"
    assert lots[1]["orange"] == 1
    assert lots[1]["roi_fail_rate"] == 100.0
    assert "LOT-001 contient 1 defaut(s) GT." in production_alerts(lots, [])


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
    assert rows[0]["activated_for_next_events"] is True
    assert rows[0]["activation_scope"] == "mlflow_registry"
    assert rows[0]["registry"] == "registered"
    assert rows[0]["promotion"] == "promoted"
    assert rows[0]["mlflow_run_id"] == "abc123"
