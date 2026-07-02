from __future__ import annotations

import json
from pathlib import Path

from scripts.export_drift_prometheus_backfill import _drift_observation_samples, _openmetrics


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_export_drift_observation_openmetrics_with_historical_timestamps(tmp_path) -> None:
    observation_dir = tmp_path / "drift_observation_001"
    observation_dir.mkdir()
    _write_json(
        observation_dir / "summary.json",
        {
            "scenario_id": "production_replay_natural_piece_b_to_piece_a_p4_drift",
            "run_id": "drift_observation_001",
            "first_confirmed_window_index": 2,
            "active_classification_runtime": {
                "version": "rd_feature_ae_gated_natural_cycle_001",
                "registered_model_version": "5",
                "registry_model_name": "feature_ae_classifier",
                "registry_stage": "test",
                "runtime_contract_status": "loaded",
            },
        },
    )
    _write_jsonl(
        observation_dir / "windows.jsonl",
        [
            {
                "evaluated_at": "2026-06-29T16:06:53+00:00",
                "status": "suspected",
                "window_index": 1,
                "drift_confirmed": False,
                "metrics": {
                    "domain_ratio": 1.0,
                    "drift_score": 1.0,
                    "degradation_score": 1.0,
                    "domain_score": 1.0,
                    "roi_mask_nn_distance": 0.74,
                    "roi_mask_novelty_rate": 1.0,
                    "roi_area_ratio": 0.50,
                    "context_events_total": 93,
                    "window_events": 30,
                    "alert_rate": 0.2,
                },
            },
            {
                "evaluated_at": "2026-06-29T16:07:36+00:00",
                "status": "confirmed",
                "window_index": 2,
                "drift_confirmed": True,
                "metrics": {
                    "domain_ratio": 1.0,
                    "drift_score": 1.0,
                    "degradation_score": 1.0,
                    "domain_score": 1.0,
                    "window_events": 30,
                    "alert_rate": 0.23,
                },
            },
        ],
    )

    body = _openmetrics(_drift_observation_samples(observation_dir))

    assert 'iqa_drift_status{observation_run_id="drift_observation_001"' in body
    assert 'status="suspected"} 1 1782749213' in body
    assert 'status="confirmed"} 1 1782749256' in body
    assert 'iqa_drift_trigger_lifecycle{observation_run_id="drift_observation_001"' in body
    assert " 1 1782749256" in body
    assert "iqa_drift_first_confirmed_window" in body
    assert "iqa_drift_roi_mask_nn_distance" in body
    assert "iqa_drift_roi_mask_novelty_rate" in body
    assert "iqa_drift_roi_area_ratio" in body
    assert "iqa_drift_context_events_total" in body
    assert "iqa_drift_active_model_info" in body
    assert body.endswith("# EOF\n")
