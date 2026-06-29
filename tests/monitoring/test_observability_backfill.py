from __future__ import annotations

import json
from pathlib import Path

from scripts import backfill_observability_from_artifacts as backfill


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_backfill_lifecycle_artifacts_pushes_epoch_and_gate_events(tmp_path) -> None:
    run_dir = tmp_path / "replay_lifecycle_001"
    candidate_dir = run_dir / "models" / "cycle_001"
    candidate_dir.mkdir(parents=True)
    _write_json(
        run_dir / "progress.json",
        {
            "scenario_id": "production_replay_natural_piece_b_full",
            "run_id": "replay_lifecycle_001",
            "events_processed": 372,
            "cycles_completed": 1,
        },
    )
    _write_json(
        run_dir / "summary.json",
        {
            "scenario_id": "production_replay_natural_piece_b_full",
            "run_id": "replay_lifecycle_001",
            "events_processed": 372,
            "active_classification_runtime_final": {
                "version": "candidate_v001",
                "registry_model_name": "feature_ae_classifier__production_replay_natural_piece_b_full",
                "registered_model_version": "4",
            },
            "active_localization_runtime_final": {
                "version": "candidate_v001",
                "registry_model_name": "feature_ae_localization__production_replay_natural_piece_b_full",
                "registered_model_version": "5",
            },
        },
    )
    _write_jsonl(
        run_dir / "cycles.jsonl",
        [
            {
                "cycle_id": "cycle_001",
                "candidate_version": "candidate_v001",
                "candidate_init_policy": "active",
                "candidate_run_dir": str(candidate_dir),
                "localization_promotion_status": "promoted",
                "classification_promotion_status": "promoted",
                "localization_metric_delta": 0.2,
                "classification_metric_delta": -1,
                "localization_active_metrics_on_eval_set": {
                    "pixel_aupimo_1e-5_1e-3": 0.1,
                    "pixel_ap": 0.2,
                },
                "localization_candidate_metrics_on_eval_set": {
                    "pixel_aupimo_1e-5_1e-3": 0.3,
                    "pixel_ap": 0.4,
                },
                "classification_active_metrics_on_eval_set": {
                    "false_negatives": 2,
                    "image_ap": 0.7,
                    "image_recall": 0.8,
                },
                "classification_candidate_metrics_on_eval_set": {
                    "false_negatives": 1,
                    "image_ap": 0.9,
                    "image_recall": 1.0,
                },
            }
        ],
    )
    _write_jsonl(
        candidate_dir / "epoch_metrics.jsonl",
        [
            {
                "epoch": 1,
                "metrics": {
                    "pixel_aupimo_1e-5_1e-3": 0.11,
                    "pixel_ap": 0.22,
                    "image_ap": 0.33,
                    "false_negatives": 2,
                },
            }
        ],
    )

    events = backfill.build_lifecycle_events(run_dir)

    assert [event["event_type"] for event in events] == [
        "epoch_completed",
        "promotion_decision",
        "run_completed",
    ]
    epoch_event = events[0]
    assert epoch_event["metrics"] == {
        "pixel_aupimo": 0.11,
        "pixel_ap": 0.22,
        "image_ap": 0.33,
        "false_negatives": 2,
    }
    promotion_event = events[1]
    assert promotion_event["metrics"]["gate_localization_active_pixel_aupimo"] == 0.1
    assert promotion_event["metrics"]["gate_classification_candidate_false_negatives"] == 1
    assert promotion_event["localization_promotion_status"] == "promoted"
    assert "candidate_run_dir" not in json.dumps(events)
    assert "checkpoint" not in json.dumps(events).lower()


def test_backfill_drift_artifacts_pushes_window_events(tmp_path) -> None:
    observation_dir = tmp_path / "drift_observation_001"
    observation_dir.mkdir()
    _write_json(
        observation_dir / "summary.json",
        {
            "scenario_id": "production_replay_natural_piece_b_to_piece_a_p4_drift",
            "first_confirmed_window_index": 6,
            "active_classification_runtime": {
                "version": "classifier_v001",
                "registry_model_name": "feature_ae_classifier__production_replay_natural_piece_b_full",
                "registered_model_version": "2",
                "registry_stage": "test",
            },
        },
    )
    _write_jsonl(
        observation_dir / "windows.jsonl",
        [
            {
                "window_index": 6,
                "status": "confirmed",
                "drift_confirmed": True,
                "metrics": {
                    "window_events": 30,
                    "drift_score": 1.0,
                    "domain_ratio": 1.0,
                    "red_rate": 0.7,
                    "unexpected_red_rate": 0.7,
                    "alert_rate": 0.7,
                },
            }
        ],
    )

    events = backfill.build_drift_events(observation_dir)

    assert len(events) == 1
    event = events[0]
    assert event["event_type"] == "window_evaluated"
    assert event["window_index"] == 6
    assert event["first_confirmed_window_index"] == 6
    assert event["trigger_lifecycle"] is True
    assert event["metrics"]["drift_score"] == 1


def test_backfill_replay_posts_best_effort(monkeypatch) -> None:
    sent: list[dict] = []

    def fake_post(event: dict, *, api_url: str, service_token: str) -> None:
        sent.append({"event": event, "api_url": api_url, "service_token": service_token})

    monkeypatch.setattr(backfill, "_post_event", fake_post)

    result = backfill.replay_events(
        [{"event_type": "epoch_completed", "scenario_id": "s", "lifecycle_run_id": "r"}],
        api_url="http://api",
        service_token="token",
        pace_seconds=0,
    )

    assert result["events_sent"] == 1
    assert sent[0]["api_url"] == "http://api"
    assert sent[0]["service_token"] == "token"
