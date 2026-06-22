from __future__ import annotations

import argparse
import csv
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
        target_stage="test",
        mode=mode,
        max_events=max_events,
        max_lots=None,
        max_cycles=None,
        lifecycle_interval=50,
        publish_minio=False,
        wait_for_gpu=False,
        no_gpu_lock=True,
        output_root=tmp_path / "out",
        device="cpu",
        batch_size=1,
        epochs=1,
        max_steps=1,
        promotion_min_delta=0.0,
        require_mlflow_registry=False,
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
        run_dir = config.output_checkpoint.parent
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metric_eval_best.json").write_text(
            json.dumps(
                {
                    "image_ap": {"value": 0.91, "epoch": 1, "checkpoint": "checkpoint_best_image_ap.pt"},
                    "pixel_aupimo_1e-5_1e-3": {
                        "value": 0.42,
                        "epoch": 1,
                        "checkpoint": "checkpoint_best_pixel_aupimo_1e-5_1e-3.pt",
                    },
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "loss_history.csv").write_text("epoch,train_loss,val_loss,lr\n1,0.2,0.3,0.001\n", encoding="utf-8")
        return {"checkpoint": str(config.output_checkpoint), "run_id": "mlflow-run-001", "run_dir": str(run_dir)}

    monkeypatch.setattr(runner, "train_feature_ae_with_mlflow_logging", fake_train)
    def fake_evaluate_progressive_model(args, model_version, checkpoint_path, evaluation_set_path, output_dir):
        value = 0.2
        if model_version.endswith("cycle_001"):
            value = 0.42
        elif model_version.endswith("cycle_002"):
            value = 0.62
        elif model_version.endswith("cycle_003"):
            value = 0.82
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics = {
            "pixel_aupimo_1e-5_1e-3": value,
            "pixel_ap": value / 3.5,
            "image_recall": 1.0,
            "false_negatives": 0,
            "orange_rate": 0.0,
        }
        (output_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "metrics": metrics,
                    "images": [
                        {"image_id": f"good_{index:03d}", "score": 0.1 + (index * 0.001), "is_defective": False}
                        for index in range(8)
                    ],
                }
            ),
            encoding="utf-8",
        )
        return {
            "model_version": model_version,
            "checkpoint_path": str(checkpoint_path),
            "metrics": metrics,
            "metrics_path": str(output_dir / "metrics.json"),
        }

    monkeypatch.setattr(runner, "evaluate_progressive_model_on_set", fake_evaluate_progressive_model)
    monkeypatch.setattr(
        runner,
        "register_run_to_model",
        lambda **kwargs: {
            "registered_model_name": f"feature_ae__{kwargs['scenario_id']}",
            "version": "1",
            "stage": kwargs["stage"],
            "alias": kwargs["stage"],
            "source_of_truth": "mlflow_registry",
        },
    )
    monkeypatch.setattr(runner, "tag_mlflow_promotion_evidence", lambda cycle: None)
    return train_calls


def test_progressive_candidate_training_resets_stale_generated_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    candidate_version = "rd_feature_ae_gated_natural_cycle_001"
    run_dir = Path(".cache/iqa/models") / candidate_version
    stale_eval_dir = run_dir / "metric_eval" / "epoch_010"
    stale_eval_dir.mkdir(parents=True)
    (run_dir / "metric_eval_best.json").write_text(
        json.dumps(
            {
                "pixel_aupimo_1e-5_1e-3": {
                    "value": 0.99,
                    "epoch": 10,
                    "checkpoint": "checkpoint_best_pixel_aupimo_1e-5_1e-3.pt",
                }
            }
        ),
        encoding="utf-8",
    )
    (stale_eval_dir / "metrics.json").write_text("{}", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("relative_path\nCasting_class1/train/good/part.jpg\n", encoding="utf-8")

    def fake_train(config, git_commit):
        del git_commit
        assert config.output_checkpoint.parent == run_dir
        assert not (run_dir / "metric_eval_best.json").exists()
        assert not stale_eval_dir.exists()
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metric_eval_best.json").write_text(
            json.dumps(
                {
                    "pixel_aupimo_1e-5_1e-3": {
                        "value": 0.1,
                        "epoch": 1,
                        "checkpoint": "checkpoint_best_pixel_aupimo_1e-5_1e-3.pt",
                    }
                }
            ),
            encoding="utf-8",
        )
        return {"checkpoint": str(config.output_checkpoint), "run_id": "run-001", "run_dir": str(run_dir)}

    monkeypatch.setattr(runner, "train_feature_ae_with_mlflow_logging", fake_train)
    monkeypatch.setattr(runner, "_git_commit", lambda: "test")

    result = runner.train_progressive_candidate(
        _args(tmp_path, scenario_id=runner.NATURAL_SCENARIO_ID, mode="progressive-train"),
        candidate_version,
        manifest,
        "feature_ae_natural_cycle_001",
    )

    assert result["run_id"] == "run-001"
    metric_best = json.loads((run_dir / "metric_eval_best.json").read_text(encoding="utf-8"))
    assert metric_best["pixel_aupimo_1e-5_1e-3"]["epoch"] == 1


def test_replay_lifecycle_cycle_selects_plan_by_scenario(tmp_path: Path, monkeypatch) -> None:
    natural = tmp_path / "natural.csv"
    drift = tmp_path / "drift.csv"
    _write_replay(natural, scenario_id=runner.NATURAL_SCENARIO_ID, rows=1)
    _write_replay(drift, scenario_id=runner.DRIFT_SCENARIO_ID, rows=1)
    monkeypatch.setattr(runner, "REPLAY_PLANS", {runner.NATURAL_SCENARIO_ID: natural, runner.DRIFT_SCENARIO_ID: drift})
    monkeypatch.setattr(runner, "ACTIVE_REPLAY_SCENARIOS", tmp_path / "missing_replay_scenarios.csv")

    assert runner.load_replay_rows(runner.NATURAL_SCENARIO_ID)[0]["scenario_id"] == runner.NATURAL_SCENARIO_ID
    assert runner.load_replay_rows(runner.DRIFT_SCENARIO_ID)[0]["scenario_id"] == runner.DRIFT_SCENARIO_ID


def test_natural_replay_active_plan_has_regular_defects() -> None:
    rows = runner.load_replay_rows(runner.NATURAL_SCENARIO_ID)
    defect_positions = [
        index
        for index, row in enumerate(rows, start=1)
        if str(row.get("is_defective", "")).lower() == "true"
    ]

    assert rows[0]["dataset_version"] == "production_replay_natural_v002"
    assert defect_positions[0] <= 60
    assert len(defect_positions) == 26
    assert max(b - a for a, b in zip(defect_positions, defect_positions[1:])) <= 35


def test_validation_gt_masks_manifest_supports_pixel_metrics() -> None:
    rows = list(csv.DictReader(runner.VALIDATION_GT_MASKS_MANIFEST.open(encoding="utf-8")))

    assert len(rows) >= 30
    assert {"image_id", "relative_path", "gt_mask_path"}.issubset(rows[0])
    assert all(row["image_id"] for row in rows)
    assert all(row["gt_mask_path"].startswith("../raw/hss-iad/") for row in rows)
    assert all(row["gt_mask_path"].endswith("_mask.png") for row in rows)


def test_original_dataset_gt_mask_contract_matches_split_semantics() -> None:
    assert runner.gt_mask_path_for_original_dataset("Casting_class1/train/good/part.jpg") == ""
    assert runner.gt_mask_path_for_original_dataset("Casting_class1/test/good/part.jpg") == ""
    assert (
        runner.gt_mask_path_for_original_dataset("Casting_class1/test/defective/part.jpg")
        == "Casting_class1/ground_truth/defective/part_mask.png"
    )


def test_decision_only_triggers_natural_without_training(tmp_path: Path, monkeypatch) -> None:
    plan = tmp_path / "natural.csv"
    _write_replay(plan, scenario_id=runner.NATURAL_SCENARIO_ID, rows=60)
    monkeypatch.setattr(runner, "REPLAY_PLANS", {runner.NATURAL_SCENARIO_ID: plan})
    monkeypatch.setattr(runner, "ACTIVE_REPLAY_SCENARIOS", tmp_path / "missing_replay_scenarios.csv")
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
    monkeypatch.setattr(runner, "ACTIVE_REPLAY_SCENARIOS", tmp_path / "missing_replay_scenarios.csv")
    train_calls = _mock_runtime(monkeypatch)

    summary = runner.run_cycle(_args(tmp_path, scenario_id=runner.NATURAL_SCENARIO_ID, mode="train-on-trigger"))

    assert summary["status"] == "trained"
    assert summary["mlflow_run_id"] == "mlflow-run-001"
    assert summary["candidate_checkpoint"].endswith("feature_ae_good_v002\\checkpoint.pt") or summary["candidate_checkpoint"].endswith("feature_ae_good_v002/checkpoint.pt")
    assert len(train_calls) == 1
    assert train_calls[0].dataset_version == "feature_ae_good_v002"


def test_progressive_decision_records_multiple_cycles(tmp_path: Path, monkeypatch) -> None:
    plan = tmp_path / "natural.csv"
    _write_replay(plan, scenario_id=runner.NATURAL_SCENARIO_ID, rows=120)
    monkeypatch.setattr(runner, "REPLAY_PLANS", {runner.NATURAL_SCENARIO_ID: plan})
    monkeypatch.setattr(runner, "ACTIVE_REPLAY_SCENARIOS", tmp_path / "missing_replay_scenarios.csv")
    train_calls = _mock_runtime(monkeypatch)
    args = _args(tmp_path, scenario_id=runner.NATURAL_SCENARIO_ID, mode="progressive-decision")
    args.max_cycles = 2

    summary = runner.run_cycle(args)

    assert summary["cycles_completed"] == 2
    assert summary["models_promoted"] == []
    assert summary["promotion_chain"] == [runner.DEFAULT_FEATURE_AE_MODEL_VERSION]
    assert train_calls == []
    cycles = (Path(summary["output_dir"]) / "cycles.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(cycles) == 2
    first_cycle = json.loads(cycles[0])
    assert first_cycle["candidate_version"] == "rd_feature_ae_gated_natural_cycle_001"
    assert first_cycle["promotion_status"] == "simulated"


def test_progressive_train_promotes_multiple_test_models(tmp_path: Path, monkeypatch) -> None:
    plan = tmp_path / "natural.csv"
    _write_replay(plan, scenario_id=runner.NATURAL_SCENARIO_ID, rows=120)
    monkeypatch.setattr(runner, "REPLAY_PLANS", {runner.NATURAL_SCENARIO_ID: plan})
    monkeypatch.setattr(runner, "ACTIVE_REPLAY_SCENARIOS", tmp_path / "missing_replay_scenarios.csv")
    train_calls = _mock_runtime(monkeypatch)
    args = _args(tmp_path, scenario_id=runner.NATURAL_SCENARIO_ID, mode="progressive-train")
    args.max_cycles = 2

    summary = runner.run_cycle(args)

    assert summary["status"] == "trained"
    assert summary["cycles_completed"] == 2
    assert summary["models_promoted"] == [
        "rd_feature_ae_gated_natural_cycle_001",
        "rd_feature_ae_gated_natural_cycle_002",
    ]
    assert summary["promotion_chain"] == [
        runner.DEFAULT_FEATURE_AE_MODEL_VERSION,
        "rd_feature_ae_gated_natural_cycle_001",
        "rd_feature_ae_gated_natural_cycle_002",
    ]
    assert len(train_calls) == 2
    assert train_calls[0].candidate_version == "rd_feature_ae_gated_natural_cycle_001"
    assert train_calls[0].dataset_version == "feature_ae_natural_cycle_001"
    assert train_calls[0].gt_masks_manifest == runner.VALIDATION_GT_MASKS_MANIFEST
    snapshot = Path(train_calls[0].manifest_path)
    assert snapshot.exists()
    assert "oracle_gt_seen_lots" in snapshot.read_text(encoding="utf-8")
    cycles = [json.loads(line) for line in (Path(summary["output_dir"]) / "cycles.jsonl").read_text(encoding="utf-8").splitlines()]
    assert cycles[0]["selected_metric"] == "pixel_aupimo_1e-5_1e-3"
    assert cycles[0]["selected_metric_value"] == 0.42
    assert cycles[0]["active_model_before"] == runner.DEFAULT_FEATURE_AE_MODEL_VERSION
    assert cycles[0]["evaluation_seen_events"] == 50
    assert cycles[0]["active_metric_value"] == 0.2
    assert cycles[0]["candidate_metric_value"] == 0.42
    assert round(cycles[0]["metric_delta"], 2) == 0.22
    assert cycles[0]["promotion_policy"] == "candidate_must_improve_active_on_same_eval_set"
    assert cycles[0]["registry_status"] == "registered"
    assert cycles[0]["val_loss"] == 0.3
    assert cycles[0]["gate_decision"] == "passed"
    assert summary["metric_history"][0]["selected_metric"] == "pixel_aupimo_1e-5_1e-3"
    assert summary["best_cycle"] == "cycle_002"
    assert summary["rejected_candidates"] == []
    assert summary["promotion_policy"] == "candidate_must_improve_active_on_same_eval_set"
    assert round(summary["comparison_history"][0]["metric_delta"], 2) == 0.22


def test_progressive_train_activates_promoted_model_for_following_events(tmp_path: Path, monkeypatch) -> None:
    plan = tmp_path / "natural.csv"
    _write_replay(plan, scenario_id=runner.NATURAL_SCENARIO_ID, rows=120)
    monkeypatch.setattr(runner, "REPLAY_PLANS", {runner.NATURAL_SCENARIO_ID: plan})
    monkeypatch.setattr(runner, "ACTIVE_REPLAY_SCENARIOS", tmp_path / "missing_replay_scenarios.csv")
    _mock_runtime(monkeypatch)
    args = _args(tmp_path, scenario_id=runner.NATURAL_SCENARIO_ID, mode="progressive-train")
    args.max_cycles = 2

    summary = runner.run_cycle(args)

    run_dir = Path(summary["output_dir"])
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    first_next_lot_event = events[50]
    assert first_next_lot_event["active_model_version"] == "rd_feature_ae_gated_natural_cycle_001"
    assert first_next_lot_event["threshold_source"].startswith("progressive_eval_good_quantiles:progressive_eval_cycle_001")
    assert first_next_lot_event["threshold_orange"] != 0.42

    cycles_path = run_dir / "cycles.jsonl"
    progress_path = run_dir / "progress.json"
    lifecycle_events_path = run_dir / "lifecycle_events.jsonl"
    assert cycles_path.exists()
    assert progress_path.exists()
    assert lifecycle_events_path.exists()
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["active_model_version"] == "rd_feature_ae_gated_natural_cycle_002"
    assert progress["phase"] == "completed"
    lifecycle_events = lifecycle_events_path.read_text(encoding="utf-8")
    assert "model_activated" in lifecycle_events
    assert "gate_passed" in lifecycle_events


def test_progressive_train_rejects_candidate_without_business_metrics(tmp_path: Path, monkeypatch) -> None:
    plan = tmp_path / "natural.csv"
    _write_replay(plan, scenario_id=runner.NATURAL_SCENARIO_ID, rows=60)
    monkeypatch.setattr(runner, "REPLAY_PLANS", {runner.NATURAL_SCENARIO_ID: plan})
    monkeypatch.setattr(runner, "ACTIVE_REPLAY_SCENARIOS", tmp_path / "missing_replay_scenarios.csv")
    _mock_runtime(monkeypatch)

    def fake_train_without_metrics(config, git_commit):
        run_dir = config.output_checkpoint.parent
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metric_eval_best.json").unlink(missing_ok=True)
        (run_dir / "loss_history.csv").write_text("epoch,train_loss,val_loss,lr\n1,0.1,0.01,0.001\n", encoding="utf-8")
        return {"checkpoint": str(config.output_checkpoint), "run_id": "mlflow-run-001", "run_dir": str(run_dir)}

    monkeypatch.setattr(runner, "train_feature_ae_with_mlflow_logging", fake_train_without_metrics)
    monkeypatch.setattr(
        runner,
        "evaluate_progressive_model_on_set",
        lambda *args, **kwargs: {
            "metrics": {},
            "metrics_path": str(kwargs["output_dir"] / "metrics.json"),
        },
    )
    args = _args(tmp_path, scenario_id=runner.NATURAL_SCENARIO_ID, mode="progressive-train")
    args.max_cycles = 1

    summary = runner.run_cycle(args)

    cycle = json.loads((Path(summary["output_dir"]) / "cycles.jsonl").read_text(encoding="utf-8"))
    assert cycle["gate_decision"] == "rejected"
    assert cycle["gate_reason"] == "rejected_missing_comparable_metric"
    assert cycle["promotion_status"] == "rejected_missing_comparable_metric"
    assert cycle["selected_metric"] is None
    assert summary["models_promoted"] == []
    assert summary["promotion_chain"] == [runner.DEFAULT_FEATURE_AE_MODEL_VERSION]
    assert summary["rejected_candidates"] == ["rd_feature_ae_gated_natural_cycle_001"]


def test_progressive_train_rejects_candidate_that_does_not_improve_active_on_same_eval_set(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan = tmp_path / "natural.csv"
    _write_replay(plan, scenario_id=runner.NATURAL_SCENARIO_ID, rows=60)
    monkeypatch.setattr(runner, "REPLAY_PLANS", {runner.NATURAL_SCENARIO_ID: plan})
    monkeypatch.setattr(runner, "ACTIVE_REPLAY_SCENARIOS", tmp_path / "missing_replay_scenarios.csv")
    _mock_runtime(monkeypatch)
    monkeypatch.setattr(
        runner,
        "evaluate_progressive_model_on_set",
        lambda args, model_version, checkpoint_path, evaluation_set_path, output_dir: {
            "model_version": model_version,
            "checkpoint_path": str(checkpoint_path),
            "metrics": {"pixel_aupimo_1e-5_1e-3": 0.9 if model_version == runner.DEFAULT_FEATURE_AE_MODEL_VERSION else 0.4},
            "metrics_path": str(output_dir / "metrics.json"),
        },
    )
    args = _args(tmp_path, scenario_id=runner.NATURAL_SCENARIO_ID, mode="progressive-train")
    args.max_cycles = 1

    summary = runner.run_cycle(args)

    cycle = json.loads((Path(summary["output_dir"]) / "cycles.jsonl").read_text(encoding="utf-8"))
    assert cycle["active_metric_value"] == 0.9
    assert cycle["candidate_metric_value"] == 0.4
    assert cycle["metric_delta"] == -0.5
    assert cycle["gate_decision"] == "rejected"
    assert cycle["gate_reason"] == "candidate_did_not_improve_active_on_same_eval_set"
    assert cycle["promotion_status"] == "rejected_no_improvement"
    assert summary["models_promoted"] == []
    assert summary["promotion_chain"] == [runner.DEFAULT_FEATURE_AE_MODEL_VERSION]


def test_progressive_train_blocks_promotion_when_false_negatives_increase(tmp_path: Path, monkeypatch) -> None:
    plan = tmp_path / "natural.csv"
    _write_replay(plan, scenario_id=runner.NATURAL_SCENARIO_ID, rows=60)
    monkeypatch.setattr(runner, "REPLAY_PLANS", {runner.NATURAL_SCENARIO_ID: plan})
    monkeypatch.setattr(runner, "ACTIVE_REPLAY_SCENARIOS", tmp_path / "missing_replay_scenarios.csv")
    _mock_runtime(monkeypatch)

    def fake_evaluate(args, model_version, checkpoint_path, evaluation_set_path, output_dir):
        value = 0.8 if model_version == runner.DEFAULT_FEATURE_AE_MODEL_VERSION else 0.9
        false_negatives = 0 if model_version == runner.DEFAULT_FEATURE_AE_MODEL_VERSION else 2
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics = {
            "pixel_aupimo_1e-5_1e-3": value,
            "pixel_ap": value / 4,
            "false_negatives": false_negatives,
            "image_recall": 1.0 if false_negatives == 0 else 0.5,
        }
        (output_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "metrics": metrics,
                    "images": [
                        {"image_id": f"good_{index:03d}", "score": 0.1 + (index * 0.001), "is_defective": False}
                        for index in range(6)
                    ],
                }
            ),
            encoding="utf-8",
        )
        return {
            "model_version": model_version,
            "checkpoint_path": str(checkpoint_path),
            "metrics": metrics,
            "metrics_path": str(output_dir / "metrics.json"),
        }

    monkeypatch.setattr(runner, "evaluate_progressive_model_on_set", fake_evaluate)
    args = _args(tmp_path, scenario_id=runner.NATURAL_SCENARIO_ID, mode="progressive-train")
    args.max_cycles = 1

    summary = runner.run_cycle(args)

    cycle = json.loads((Path(summary["output_dir"]) / "cycles.jsonl").read_text(encoding="utf-8"))
    assert cycle["metric_delta"] > 0
    assert cycle["gate_decision"] == "rejected"
    assert cycle["gate_reason"] == "candidate_increases_false_negatives"
    assert cycle["promotion_status"] == "rejected_operational_guardrail"
    assert summary["promotion_chain"] == [runner.DEFAULT_FEATURE_AE_MODEL_VERSION]


def test_register_promoted_cycle_records_registry_failure_without_strict_mode(tmp_path: Path, monkeypatch) -> None:
    args = _args(tmp_path, scenario_id=runner.NATURAL_SCENARIO_ID, mode="progressive-train")
    state = runner.CycleState(
        scenario_id=runner.NATURAL_SCENARIO_ID,
        mode="progressive-train",
        run_id="run",
        output_dir=tmp_path,
    )
    monkeypatch.setattr(
        runner,
        "register_run_to_model",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("registry offline")),
    )

    result = runner.register_promoted_cycle(args, state, {"mlflow_run_id": "mlflow-001"})

    assert result["registry_status"] == "failed"
    assert result["registered_model_name"] == "feature_ae__production_replay_natural"


def test_register_promoted_cycle_raises_registry_failure_in_strict_mode(tmp_path: Path, monkeypatch) -> None:
    import pytest

    args = _args(tmp_path, scenario_id=runner.NATURAL_SCENARIO_ID, mode="progressive-train")
    args.require_mlflow_registry = True
    state = runner.CycleState(
        scenario_id=runner.NATURAL_SCENARIO_ID,
        mode="progressive-train",
        run_id="run",
        output_dir=tmp_path,
    )
    monkeypatch.setattr(
        runner,
        "register_run_to_model",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("registry offline")),
    )

    with pytest.raises(RuntimeError, match="registry offline"):
        runner.register_promoted_cycle(args, state, {"mlflow_run_id": "mlflow-001"})


def test_drift_cycle_triggers_on_confirmed_drift(tmp_path: Path, monkeypatch) -> None:
    plan = tmp_path / "drift.csv"
    _write_replay(plan, scenario_id=runner.DRIFT_SCENARIO_ID, rows=1)
    monkeypatch.setattr(runner, "REPLAY_PLANS", {runner.DRIFT_SCENARIO_ID: plan})
    monkeypatch.setattr(runner, "ACTIVE_REPLAY_SCENARIOS", tmp_path / "missing_replay_scenarios.csv")
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
