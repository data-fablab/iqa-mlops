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
            "candidate_version": "rd_feature_ae_gated_natural_cycle_001",
            "selected_metric": "image_ap",
            "selected_metric_value": 0.91,
            "gate_decision": "passed",
            "registry_stage": "test",
            "mlflow_run_id": "run-001",
        },
        {
            "cycle_id": "cycle_002",
            "candidate_version": "rd_feature_ae_gated_natural_cycle_002",
            "selected_metric": "pixel_aupimo_1e-5_1e-3",
            "selected_metric_value": 0.42,
            "gate_decision": "rejected",
            "registry_stage": "test",
            "mlflow_run_id": "run-002",
        },
    ]
    (run_dir / "cycles.jsonl").write_text(
        "".join(json.dumps(cycle) + "\n" for cycle in cycles),
        encoding="utf-8",
    )

    report = render_report(run_dir)

    assert "cycle" in report
    assert "rd_feature_ae_gated_natural_cycle_001" in report
    assert "image_ap" in report
    assert "0.91" in report
    assert "pixel_aupimo_1e-5_1e-3" in report
    assert "run-002" in report


def test_progressive_lifecycle_report_fails_without_cycles_jsonl(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="cycles.jsonl"):
        render_report(tmp_path / "missing")
