from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.progressive_lifecycle_report import render_report


def test_progressive_lifecycle_report_renders_cycle_metrics(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cycles = [
        {
            "cycle_id": "cycle_001",
            "active_model_before": "bootstrap",
            "candidate_version": "rd_feature_ae_gated_natural_cycle_001",
            "selected_metric": "image_ap",
            "selected_metric_value": 0.91,
            "evaluation_seen_events": 60,
            "active_metric_value": 0.80,
            "candidate_metric_value": 0.91,
            "metric_delta": 0.11,
            "active_false_negatives": 1,
            "candidate_false_negatives": 0,
            "active_good_red_count": 2,
            "candidate_good_red_count": 2,
            "fn_delta": -1,
            "good_red_delta": 0,
            "gate_decision": "passed",
            "gate_reason": "candidate_passed_reference_gate",
            "registry_stage": "test",
            "registered_model_version": "1",
            "mlflow_run_id": "run-001",
            "mlflow_dataset_logged": True,
            "mlflow_model_logged": True,
            "candidate_metrics_on_eval_set": {
                "pixel_ap": 0.12,
                "alert_rate": 0.2,
                "good_alert_rate": 0.1,
                "good_red_rate": 0.0,
                "false_negatives": 0,
            },
            "candidate_aupimo_stability": {"aupimo_unstable": False},
            "cache_status": "miss_stored",
            "cache_hit": False,
            "epoch_metric_history": [
                {
                    "epoch": 1,
                    "metrics": {"pixel_aupimo_1e-5_1e-3": 0.91, "pixel_ap": 0.12},
                    "predictions_path": "metric_eval/epoch_001/predictions.npz",
                }
            ],
        },
        {
            "cycle_id": "cycle_002",
            "active_model_before": "rd_feature_ae_gated_natural_cycle_001",
            "candidate_version": "rd_feature_ae_gated_natural_cycle_002",
            "selected_metric": "pixel_aupimo_1e-5_1e-3",
            "selected_metric_value": 0.42,
            "evaluation_seen_events": 120,
            "active_metric_value": 0.50,
            "candidate_metric_value": 0.42,
            "metric_delta": -0.08,
            "active_false_negatives": 0,
            "candidate_false_negatives": 2,
            "active_good_red_count": 1,
            "candidate_good_red_count": 4,
            "fn_delta": 2,
            "good_red_delta": 3,
            "gate_decision": "rejected",
            "gate_reason": "candidate_increases_false_negatives",
            "registry_stage": "test",
            "registry_status": "not_registered",
            "mlflow_run_id": "run-002",
            "mlflow_dataset_logged": False,
            "mlflow_model_logged": False,
        },
    ]
    (run_dir / "cycles.jsonl").write_text(
        "".join(json.dumps(cycle) + "\n" for cycle in cycles),
        encoding="utf-8",
    )

    report = render_report(run_dir, show_epochs=True, show_cache=True, show_mlflow=True)

    assert "cycle" in report
    assert "active_before" in report
    assert "rd_feature_ae_gated_natural_cycle_001" in report
    assert "0.91" in report
    assert "+0.11" in report
    assert "-0.08" in report
    assert "test:v1" in report
    assert "not_registered" in report
    assert "miss_stored" in report
    assert "run_id" in report
    assert "active_fn" in report
    assert "candidate_good_red" in report
    assert "candidate_passed_reference_gate" in report
    assert "run-001" in report
    assert "yes" in report
    assert "no" in report
    assert "epoch metrics" in report


def test_progressive_lifecycle_report_fails_without_cycles_jsonl(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="cycles.jsonl"):
        render_report(tmp_path / "missing")


def test_progressive_lifecycle_report_reads_in_progress_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "progress.json").write_text(
        json.dumps(
            {
                "phase": "cycle_running",
                "active_model_version": "rd_feature_ae_gated_v001_bootstrap",
                "events_processed": 120,
                "lots_processed": 4,
            }
        ),
        encoding="utf-8",
    )

    report = render_report(run_dir)

    assert "No completed cycles yet" in report
    assert "cycle_running" in report
    assert "rd_feature_ae_gated_v001_bootstrap" in report
