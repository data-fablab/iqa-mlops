from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from scripts import run_replay_lifecycle_cycle as runner
from iqa.storage.object_store import InMemoryObjectStore


def _args(tmp_path: Path, *, scenario_id: str, mode: str = "decision-only", max_events: int | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        scenario_id=scenario_id,
        image_root=tmp_path / "images",
        stage="test",
        mode=mode,
        max_events=max_events,
        max_lots=None,
        publish_minio=False,
        wait_for_gpu=False,
        no_gpu_lock=True,
        output_root=tmp_path / "out",
        device="cpu",
        batch_size=1,
        epochs=1,
        max_steps=1,
    )


def _write_replay(path: Path, *, scenario_id: str, rows: int = 60) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "event_id,piece_event_id,scenario_id,lot_id,source_class,dataset_version,relative_paths,image_ids,is_defective,has_mask"
    ]
    for index in range(rows):
        lot_id = "LOT-001" if index < 50 else "LOT-002"
        lines.append(
            f"event_{index:03d},piece_{index:03d},{scenario_id},{lot_id},Casting_class1,"
            f"{scenario_id}_v001,Casting_class1/train/good/part_{index:03d}.jpg,img_{index:03d},false,false"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mock_runtime(monkeypatch) -> list[argparse.Namespace]:
    train_calls: list[argparse.Namespace] = []
    monkeypatch.setattr(runner, "resolve_roi_segmenter_checkpoint", lambda *args, **kwargs: Path("roi.pt"))
    monkeypatch.setattr(runner, "resolve_feature_ae_checkpoint", lambda *args, **kwargs: Path("feature.pt"))
    monkeypatch.setattr(runner, "create_visual_object_store", lambda: InMemoryObjectStore())
    monkeypatch.setattr(
        runner,
        "load_feature_ae_decision_thresholds",
        lambda *args, **kwargs: {
            "method": "calibration_good_quantiles",
            "threshold_orange": 0.42,
            "threshold_red": 0.84,
        },
    )
    monkeypatch.setattr(
        runner,
        "predict_roi_image",
        lambda *args, **kwargs: SimpleNamespace(roi_quality_status="ok", roi_ratio=0.42),
    )
    monkeypatch.setattr(
        runner,
        "predict_feature_ae_image",
        lambda *args, **kwargs: SimpleNamespace(
            status="green",
            score=0.01,
            threshold_orange=kwargs["threshold_orange"],
            threshold_red=kwargs["threshold_red"],
            threshold_source=kwargs["threshold_source"],
        ),
    )

    def fake_train(config, git_commit):
        train_calls.append(config)
        return {"checkpoint": str(config.output_checkpoint), "run_id": "mlflow-run-001"}

    monkeypatch.setattr(runner, "train_feature_ae_with_mlflow_logging", fake_train)
    return train_calls


def test_replay_lifecycle_cycle_selects_plan_by_scenario(tmp_path: Path, monkeypatch) -> None:
    natural = tmp_path / "natural.csv"
    drift = tmp_path / "drift.csv"
    _write_replay(natural, scenario_id=runner.NATURAL_SCENARIO_ID, rows=1)
    _write_replay(drift, scenario_id=runner.DRIFT_SCENARIO_ID, rows=1)
    monkeypatch.setattr(runner, "REPLAY_PLANS", {runner.NATURAL_SCENARIO_ID: natural, runner.DRIFT_SCENARIO_ID: drift})

    assert runner.load_replay_rows(runner.NATURAL_SCENARIO_ID)[0]["scenario_id"] == runner.NATURAL_SCENARIO_ID
    assert runner.load_replay_rows(runner.DRIFT_SCENARIO_ID)[0]["scenario_id"] == runner.DRIFT_SCENARIO_ID


def test_decision_only_triggers_natural_without_training(tmp_path: Path, monkeypatch) -> None:
    plan = tmp_path / "natural.csv"
    _write_replay(plan, scenario_id=runner.NATURAL_SCENARIO_ID, rows=60)
    monkeypatch.setattr(runner, "REPLAY_PLANS", {runner.NATURAL_SCENARIO_ID: plan})
    train_calls = _mock_runtime(monkeypatch)

    summary = runner.run_cycle(_args(tmp_path, scenario_id=runner.NATURAL_SCENARIO_ID, mode="decision-only"))

    assert summary["trigger_lifecycle"] is True
    assert summary["trigger_reason"] == "natural_50_oracle_conformes"
    assert summary["candidate_dataset_version"] == "feature_ae_good_v002"
    assert summary["status"] == "validated"
    assert train_calls == []
    lots = (Path(summary["output_dir"]) / "lots.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lots[0])["conforming_validated_count"] == 50
    events = (Path(summary["output_dir"]) / "events.jsonl").read_text(encoding="utf-8").splitlines()
    first_event = json.loads(events[0])
    assert first_event["threshold_orange"] == 0.42
    assert first_event["threshold_red"] == 0.84
    assert first_event["threshold_source"] == "manifest:calibration_good_quantiles"
    assert first_event["roi_mask_path"].endswith("_roi.png")
    assert first_event["heatmap_path"].endswith("_heatmap.png")
    assert "roi_mask_uri" in first_event
    assert "heatmap_uri" in first_event


def test_train_on_trigger_trains_candidate_once(tmp_path: Path, monkeypatch) -> None:
    plan = tmp_path / "natural.csv"
    _write_replay(plan, scenario_id=runner.NATURAL_SCENARIO_ID, rows=60)
    monkeypatch.setattr(runner, "REPLAY_PLANS", {runner.NATURAL_SCENARIO_ID: plan})
    train_calls = _mock_runtime(monkeypatch)

    summary = runner.run_cycle(_args(tmp_path, scenario_id=runner.NATURAL_SCENARIO_ID, mode="train-on-trigger"))

    assert summary["status"] == "trained"
    assert summary["mlflow_run_id"] == "mlflow-run-001"
    assert summary["candidate_checkpoint"].endswith("feature_ae_good_v002\\checkpoint.pt") or summary["candidate_checkpoint"].endswith("feature_ae_good_v002/checkpoint.pt")
    assert len(train_calls) == 1
    assert train_calls[0].dataset_version == "feature_ae_good_v002"


def test_drift_cycle_triggers_on_confirmed_drift(tmp_path: Path, monkeypatch) -> None:
    plan = tmp_path / "drift.csv"
    _write_replay(plan, scenario_id=runner.DRIFT_SCENARIO_ID, rows=1)
    monkeypatch.setattr(runner, "REPLAY_PLANS", {runner.DRIFT_SCENARIO_ID: plan})
    _mock_runtime(monkeypatch)

    summary = runner.run_cycle(_args(tmp_path, scenario_id=runner.DRIFT_SCENARIO_ID, mode="decision-only"))

    assert summary["trigger_lifecycle"] is True
    assert summary["trigger_reason"] == "drift_confirmed"
    assert summary["candidate_dataset_version"] == "feature_ae_good_v003"


def test_runtime_thresholds_fall_back_to_legacy_defaults(monkeypatch) -> None:
    monkeypatch.setattr(runner, "load_feature_ae_decision_thresholds", lambda *args, **kwargs: None)

    thresholds = runner.resolve_runtime_thresholds("missing_thresholds")

    assert thresholds == {
        "threshold_orange": 0.02,
        "threshold_red": 0.05,
        "threshold_source": "legacy_default",
    }
