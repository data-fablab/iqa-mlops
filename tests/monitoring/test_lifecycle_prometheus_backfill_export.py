from __future__ import annotations

import json
from pathlib import Path

from scripts.export_lifecycle_prometheus_backfill import _openmetrics, _promotion_selection_samples


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_export_lifecycle_promotion_selection_openmetrics_with_historical_timestamps(tmp_path) -> None:
    run_dir = tmp_path / "replay_lifecycle_001"
    candidate_dir = tmp_path / "models" / "candidate"
    candidate_dir.mkdir(parents=True)
    run_dir.mkdir()
    _write_json(
        run_dir / "progress.json",
        {
            "scenario_id": "production_replay_natural_piece_b_full",
            "run_id": "replay_lifecycle_001",
        },
    )
    _write_json(
        candidate_dir / "metric_eval_best.json",
        {
            "image_ap": {
                "epoch": 2,
                "value": 0.91,
            }
        },
    )
    _write_jsonl(
        run_dir / "lifecycle_events.jsonl",
        [
            {
                "event_type": "dual_gate_decision",
                "cycle_id": "cycle_001",
                "timestamp": "2026-06-26T21:40:00+00:00",
            }
        ],
    )
    _write_jsonl(
        run_dir / "cycles.jsonl",
        [
            {
                "cycle_id": "cycle_001",
                "candidate_version": "candidate_v001",
                "candidate_run_dir": str(candidate_dir),
                "localization_promotion_status": "promoted",
                "localization_selected_metric": "pixel_aupimo_1e-5_1e-3",
                "localization_candidate_metric_value": 0.42,
                "selected_epoch": 4,
                "classification_promotion_status": "promoted",
                "classification_candidate_checkpoint": str(candidate_dir / "checkpoint_best_image.pt"),
                "classification_selected_metric": "false_negatives",
                "classification_candidate_metric_value": 1,
                "classification_gate": {
                    "active_false_negatives": 1,
                    "candidate_false_negatives": 0,
                    "active_good_red_count": 1,
                    "candidate_good_red_count": 1,
                    "good_red_delta": 0,
                },
                "localization_gate": {
                    "active_value": 0.18,
                    "candidate_value": 0.23,
                    "metric": "pixel_aupimo_1e-5_1e-3",
                },
                "localization_active_metrics_on_eval_set": {
                    "pixel_aupimo_1e-5_1e-3": 0.18,
                    "pixel_ap": 0.15,
                },
                "localization_candidate_metrics_on_eval_set": {
                    "pixel_aupimo_1e-5_1e-3": 0.23,
                    "pixel_ap": 0.19,
                },
                "piece_b_non_regression_classification_active_metrics": {
                    "false_negatives": 5,
                    "image_ap": 0.74,
                },
                "piece_b_non_regression_classification_candidate_metrics": {
                    "false_negatives": 6,
                    "image_ap": 0.7,
                },
                "training_manifest_stats": {
                    "anchor_good_count": 2,
                    "seen_conforming_count": 72,
                    "total_count": 74,
                },
            }
        ],
    )

    samples = _promotion_selection_samples(run_dir)
    body = _openmetrics(samples)

    assert 'iqa_lifecycle_promotion_selected_epoch{candidate_version="candidate_v001"' in body
    assert 'role="localization"' in body
    assert 'role="classification"' in body
    assert " 4 1782510000" in body
    assert " 2 1782510000" in body
    assert "iqa_lifecycle_promotion_selected_metric_value" in body
    assert 'iqa_lifecycle_train_set_size{candidate_init_policy=""' in body
    assert 'kind="total"' in body
    assert " 74 1782510000" in body
    assert 'kind="seen_conforming"' in body
    assert " 72 1782510000" in body
    assert 'kind="anchor_good"' in body
    assert 'metric="good_red_count"' in body
    assert 'metric="pixel_ap"' in body
    assert 'metric="piece_b_false_negatives"' in body
    assert 'metric="piece_b_image_ap"' in body
    assert body.endswith("# EOF\n")
