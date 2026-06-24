from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_replay_lifecycle_cycle as runner
from iqa.storage.object_store import InMemoryObjectStore


def _args(tmp_path: Path, *, scenario_id: str, mode: str = "decision-only", max_events: int | None = None) -> argparse.Namespace:
    anchor_good = tmp_path / "anchor_good.csv"
    if not anchor_good.exists():
        anchor_good.write_text(
            "image_id,image_ids,relative_path,relative_paths,event_id,source_class,split_set,label,is_defective,scenario_id,dataset_version,manifest_version,gt_mask_path,oracle_verdict,train_eligible,train_eligibility_source,quarantine_reason\n"
            "anchor_001,anchor_001,Casting_class1/train/good/anchor_001.jpg,Casting_class1/train/good/anchor_001.jpg,anchor_event_001,Casting_class1,anchor,good,false,"
            f"{scenario_id},anchor_good,anchor_good_v001,,conforme,true,anchor_good_reference,\n",
            encoding="utf-8",
        )
    reference_eval = tmp_path / "reference_eval.csv"
    if not reference_eval.exists():
        reference_eval.write_text(
            "image_id,image_ids,relative_path,relative_paths,event_id,piece_event_id,lot_id,source_class,split_set,label,is_defective,scenario_id,dataset_version,manifest_version,gt_mask_path,oracle_verdict,train_eligible,train_eligibility_source,quarantine_reason,roi_mask_path,roi_probability_path\n"
            "ref_001,ref_001,Casting_class1/test/good/ref_001.jpg,Casting_class1/test/good/ref_001.jpg,ref_event_001,ref_piece_001,REF-LOT,Casting_class1,test,good,false,"
            f"{scenario_id},reference_eval,reference_eval_v001,,conforme,false,reference_eval,,,\n",
            encoding="utf-8",
        )
    reference_gt_masks = tmp_path / "reference_gt_masks.csv"
    reference_gt_masks.write_text("image_id,gt_mask_path\n", encoding="utf-8")
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
        gate_eval_profile="fast",
        promotion_min_delta=0.0,
        require_mlflow_registry=False,
        anchor_good_manifest=anchor_good,
        anchor_good_max_per_class=2,
        reference_eval_manifest=reference_eval,
        reference_gt_masks_manifest=reference_gt_masks,
        max_good_red_regression=1,
        candidate_init_policy="fresh",
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
    def fake_evaluate_progressive_model(
        args,
        model_version,
        checkpoint_path,
        evaluation_set_path,
        output_dir,
        **kwargs,
    ):
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

    monkeypatch.setattr(runner, "evaluate_reference_model_on_set", fake_evaluate_progressive_model)
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
    assert summary["candidate_dataset_version"] == "feature_ae_good_mvp_v001"
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
    assert summary["candidate_checkpoint"].endswith("feature_ae_good_mvp_v001\\checkpoint.pt") or summary["candidate_checkpoint"].endswith("feature_ae_good_mvp_v001/checkpoint.pt")
    assert len(train_calls) == 1
    assert train_calls[0].dataset_version == "feature_ae_good_mvp_v001"


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
    assert cycles[0]["promotion_policy"] == runner.PROGRESSIVE_PROMOTION_POLICY
    assert cycles[0]["registry_status"] == "registered"
    assert cycles[0]["val_loss"] == 0.3
    assert cycles[0]["gate_decision"] == "passed"
    assert cycles[0]["localization_gate"]["passed"] is True
    assert cycles[0]["localization_gate"]["metric"] == "pixel_aupimo_1e-5_1e-3"
    assert cycles[0]["classification_gate"]["passed"] is True
    assert cycles[0]["classification_progress"]["non_regression"] is True
    assert summary["metric_history"][0]["selected_metric"] == "pixel_aupimo_1e-5_1e-3"
    assert summary["best_cycle"] == "cycle_002"
    assert summary["rejected_candidates"] == []
    assert summary["promotion_policy"] == runner.PROGRESSIVE_PROMOTION_POLICY
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
    assert first_next_lot_event["threshold_source"].startswith("panel_good_quantiles:reference_eval")
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
        "evaluate_reference_model_on_set",
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
        "evaluate_reference_model_on_set",
        lambda args, model_version, checkpoint_path, evaluation_set_path, output_dir, **kwargs: {
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
    assert cycle["gate_reason"] == "candidate_regressed_on_reference_panel"
    assert cycle["promotion_status"] == "rejected_reference_regression"
    assert cycle["localization_gate"]["passed"] is False
    assert cycle["classification_gate"]["passed"] is True
    assert summary["models_promoted"] == []
    assert summary["promotion_chain"] == [runner.DEFAULT_FEATURE_AE_MODEL_VERSION]


def test_progressive_train_promotes_reference_win_despite_progressive_metric_regression(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan = tmp_path / "natural.csv"
    _write_replay(plan, scenario_id=runner.NATURAL_SCENARIO_ID, rows=60)
    monkeypatch.setattr(runner, "REPLAY_PLANS", {runner.NATURAL_SCENARIO_ID: plan})
    monkeypatch.setattr(runner, "ACTIVE_REPLAY_SCENARIOS", tmp_path / "missing_replay_scenarios.csv")
    _mock_runtime(monkeypatch)

    def fake_evaluate(args, model_version, checkpoint_path, evaluation_set_path, output_dir, **kwargs):
        panel = "reference" if "reference" in {part.lower() for part in output_dir.parts} else "progressive"
        value = 0.5
        if model_version.endswith("cycle_001"):
            value = 0.7 if panel == "reference" else 0.3
        false_negatives = 0 if model_version.endswith("cycle_001") else 1
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

    monkeypatch.setattr(runner, "evaluate_reference_model_on_set", fake_evaluate)
    args = _args(tmp_path, scenario_id=runner.NATURAL_SCENARIO_ID, mode="progressive-train")
    args.max_cycles = 1

    summary = runner.run_cycle(args)

    cycle = json.loads((Path(summary["output_dir"]) / "cycles.jsonl").read_text(encoding="utf-8"))
    assert cycle["reference_metric_delta"] > 0
    assert "progressive_metric_delta" not in cycle
    assert cycle["gate_decision"] == "passed"
    assert cycle["gate_reason"] == "candidate_passed_representative_validation_gate"
    assert cycle["promotion_status"] == "promoted"
    assert cycle["localization_gate"]["passed"] is True
    assert cycle["classification_gate"]["passed"] is True
    assert cycle["classification_progress"]["improved"] is True
    assert cycle["classification_progress"]["non_regression"] is True
    assert cycle["classification_gate"]["fn_delta"] == -1
    assert cycle["classification_gate"]["image_recall_delta"] == 0.5
    assert summary["promotion_chain"] == [
        runner.DEFAULT_FEATURE_AE_MODEL_VERSION,
        "rd_feature_ae_gated_natural_cycle_001",
    ]


def test_progressive_train_blocks_promotion_when_false_negatives_increase(tmp_path: Path, monkeypatch) -> None:
    plan = tmp_path / "natural.csv"
    _write_replay(plan, scenario_id=runner.NATURAL_SCENARIO_ID, rows=60)
    monkeypatch.setattr(runner, "REPLAY_PLANS", {runner.NATURAL_SCENARIO_ID: plan})
    monkeypatch.setattr(runner, "ACTIVE_REPLAY_SCENARIOS", tmp_path / "missing_replay_scenarios.csv")
    _mock_runtime(monkeypatch)

    def fake_evaluate(args, model_version, checkpoint_path, evaluation_set_path, output_dir, **kwargs):
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

    monkeypatch.setattr(runner, "evaluate_reference_model_on_set", fake_evaluate)
    args = _args(tmp_path, scenario_id=runner.NATURAL_SCENARIO_ID, mode="progressive-train")
    args.max_cycles = 1

    summary = runner.run_cycle(args)

    cycle = json.loads((Path(summary["output_dir"]) / "cycles.jsonl").read_text(encoding="utf-8"))
    assert cycle["metric_delta"] > 0
    assert cycle["gate_decision"] == "rejected"
    assert cycle["gate_reason"] == "candidate_increases_false_negatives"
    assert cycle["promotion_status"] == "rejected_operational_guardrail"
    assert cycle["localization_gate"]["passed"] is True
    assert cycle["classification_gate"]["passed"] is False
    assert cycle["classification_gate"]["fn_delta"] == 2
    assert cycle["classification_progress"]["non_regression"] is False
    assert summary["promotion_chain"] == [runner.DEFAULT_FEATURE_AE_MODEL_VERSION]


def test_progressive_train_blocks_candidate_that_increases_good_red_count(tmp_path: Path, monkeypatch) -> None:
    plan = tmp_path / "natural.csv"
    _write_replay(plan, scenario_id=runner.NATURAL_SCENARIO_ID, rows=60)
    monkeypatch.setattr(runner, "REPLAY_PLANS", {runner.NATURAL_SCENARIO_ID: plan})
    monkeypatch.setattr(runner, "ACTIVE_REPLAY_SCENARIOS", tmp_path / "missing_replay_scenarios.csv")
    _mock_runtime(monkeypatch)

    def fake_evaluate(args, model_version, checkpoint_path, evaluation_set_path, output_dir, **kwargs):
        value = 0.4 if model_version == runner.DEFAULT_FEATURE_AE_MODEL_VERSION else 0.9
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics = {
            "pixel_aupimo_1e-5_1e-3": value,
            "pixel_ap": value / 4,
            "false_negatives": 0,
            "image_recall": 1.0,
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

    monkeypatch.setattr(runner, "evaluate_reference_model_on_set", fake_evaluate)
    monkeypatch.setattr(
        runner,
        "thresholds_from_evaluation_scores",
        lambda payload, model_version, role, evaluation_set_id: {
            "method": "test",
            "threshold_orange": 999.0 if role == "active_before" else 0.0,
            "threshold_red": 999.0 if role == "active_before" else 0.0,
            "threshold_source": f"test:{role}",
            "calibration_signature": f"test:{role}",
        },
    )
    args = _args(tmp_path, scenario_id=runner.NATURAL_SCENARIO_ID, mode="progressive-train")
    args.max_cycles = 1

    summary = runner.run_cycle(args)

    cycle = json.loads((Path(summary["output_dir"]) / "cycles.jsonl").read_text(encoding="utf-8"))
    assert cycle["metric_delta"] > 0
    assert cycle["good_red_delta"] > args.max_good_red_regression
    assert cycle["gate_decision"] == "rejected"
    assert cycle["gate_reason"] == "candidate_increases_good_red_count"
    assert cycle["promotion_status"] == "rejected_operational_guardrail"
    assert cycle["localization_gate"]["passed"] is True
    assert cycle["classification_gate"]["passed"] is False
    assert cycle["classification_gate"]["good_red_delta"] > args.max_good_red_regression
    assert cycle["classification_progress"]["non_regression"] is False
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


def test_register_promoted_cycle_records_missing_mlflow_model_without_strict_mode(tmp_path: Path, monkeypatch) -> None:
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
        lambda **kwargs: (_ for _ in ()).throw(
            FileNotFoundError("missing_mlflow_model_artifact: run mlflow-001 has no model/MLmodel artifact")
        ),
    )

    result = runner.register_promoted_cycle(args, state, {"mlflow_run_id": "mlflow-001"})

    assert result["registry_status"] == "failed_missing_mlflow_model"
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


def test_tag_mlflow_promotion_evidence_tags_run_and_model_version(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, list[tuple[str, ...] | tuple[str, str, float] | tuple[str, str, str | None]]] = {
        "run_tags": [],
        "version_tags": [],
        "descriptions": [],
        "params": [],
        "metrics": [],
        "artifacts": [],
    }
    active_metrics = tmp_path / "active_metrics.json"
    active_metrics.write_text(json.dumps({"metrics": {"pixel_aupimo_1e-5_1e-3": 0.3}}), encoding="utf-8")
    candidate_metrics = tmp_path / "candidate_metrics.json"
    candidate_metrics.write_text(json.dumps({"metrics": {"pixel_aupimo_1e-5_1e-3": 0.4}}), encoding="utf-8")

    class FakeClient:
        def set_tag(self, run_id: str, key: str, value: str) -> None:
            calls["run_tags"].append((run_id, key, value))

        def log_param(self, run_id: str, key: str, value: str) -> None:
            calls["params"].append((run_id, key, value))

        def log_metric(self, run_id: str, key: str, value: float) -> None:
            calls["metrics"].append((run_id, key, value))

        def log_artifact(self, run_id: str, local_path: str, artifact_path: str | None = None) -> None:
            calls["artifacts"].append((run_id, Path(local_path).name, artifact_path))

        def set_model_version_tag(self, name: str, version: str, key: str, value: str) -> None:
            calls["version_tags"].append((name, version, key, value))

        def update_model_version(self, name: str, version: str, description: str) -> None:
            calls["descriptions"].append((name, version, description))

    class FakeTracking:
        MlflowClient = FakeClient

    class FakeMlflow:
        tracking = FakeTracking()

    monkeypatch.setitem(sys.modules, "mlflow", FakeMlflow())

    runner.tag_mlflow_promotion_evidence(
        {
            "mlflow_run_id": "run-001",
            "registered_model_name": "feature_ae__production_replay_natural",
            "registered_model_version": "4",
            "candidate_version": "rd_feature_ae_gated_natural_cycle_001",
            "gate_decision": "passed",
            "gate_reason": "candidate_improved_on_reference_panel",
            "gate_eval_profile": "fast",
            "promotion_status": "promoted",
            "selected_metric": "pixel_aupimo_1e-5_1e-3",
            "active_metric_value": 0.3,
            "candidate_metric_value": 0.4,
            "metric_delta": 0.1,
            "active_false_negatives": 1,
            "candidate_false_negatives": 0,
            "fn_delta": -1,
            "active_good_red_count": 2,
            "candidate_good_red_count": 2,
            "good_red_delta": 0,
            "active_metrics_on_eval_set": {
                "pixel_aupimo_1e-5_1e-3": 0.3,
                "false_negatives": 1,
                "good_red_count": 2,
            },
            "candidate_metrics_on_eval_set": {
                "pixel_aupimo_1e-5_1e-3": 0.4,
                "false_negatives": 0,
                "good_red_count": 2,
            },
            "mvp_gate": {
                "metric_ok": True,
                "false_negatives_ok": True,
                "good_red_ok": True,
                "thresholds_ok": True,
            },
            "localization_gate": {
                "metric": "pixel_aupimo_1e-5_1e-3",
                "active_value": 0.3,
                "candidate_value": 0.4,
                "delta": 0.1,
                "passed": True,
            },
            "classification_gate": {
                "active_false_negatives": 1,
                "candidate_false_negatives": 0,
                "fn_delta": -1,
                "active_image_recall": 0.5,
                "candidate_image_recall": 1.0,
                "image_recall_delta": 0.5,
                "active_good_red_count": 2,
                "candidate_good_red_count": 2,
                "good_red_delta": 0,
                "passed": True,
            },
            "classification_progress": {
                "improved": True,
                "non_regression": True,
                "summary": "improved: FN 1 -> 0",
            },
            "classification_progress_improved": True,
            "classification_progress_summary": "improved: FN 1 -> 0",
            "active_eval_metrics_path": str(active_metrics),
            "candidate_eval_metrics_path": str(candidate_metrics),
        }
    )

    assert ("run-001", "gate_decision", "passed") in calls["run_tags"]
    assert ("run-001", "gate_eval_profile", "fast") in calls["run_tags"]
    assert ("run-001", "gate.decision", "passed") in calls["params"]
    assert ("run-001", "gate.eval_profile", "fast") in calls["params"]
    assert ("run-001", "gate.passed", 1.0) in calls["metrics"]
    assert ("run-001", "gate.metric_delta", 0.1) in calls["metrics"]
    assert ("run-001", "gate.candidate.pixel_aupimo_1e-5_1e-3", 0.4) in calls["metrics"]
    assert ("run-001", "gate.false_negatives_ok", 1.0) in calls["metrics"]
    assert ("run-001", "gate.localization.passed", 1.0) in calls["metrics"]
    assert ("run-001", "gate.localization.delta", 0.1) in calls["metrics"]
    assert ("run-001", "gate.classification.passed", 1.0) in calls["metrics"]
    assert ("run-001", "gate.classification.fn_delta", -1.0) in calls["metrics"]
    assert ("run-001", "gate.classification.image_recall_delta", 0.5) in calls["metrics"]
    assert ("run-001", "gate.classification_progress.improved", 1.0) in calls["metrics"]
    assert ("run-001", "classification_progress.improved", "True") in calls["params"]
    assert ("run-001", "gate_evidence.json", "gate") in calls["artifacts"]
    assert ("run-001", "active_metrics.json", "gate/active") in calls["artifacts"]
    assert ("run-001", "candidate_metrics.json", "gate/candidate") in calls["artifacts"]
    assert (
        "feature_ae__production_replay_natural",
        "4",
        "gate_decision",
        "passed",
    ) in calls["version_tags"]
    assert calls["descriptions"] == [
        (
            "feature_ae__production_replay_natural",
            "4",
            "rd_feature_ae_gated_natural_cycle_001: gate=passed promotion=promoted "
            "metric=pixel_aupimo_1e-5_1e-3 delta=0.1 fn=0 good_red=2",
        )
    ]


def test_tag_mlflow_promotion_evidence_logs_rejected_gate_without_registry(monkeypatch) -> None:
    calls: dict[str, list[tuple[str, ...] | tuple[str, str, float]]] = {
        "run_tags": [],
        "version_tags": [],
        "metrics": [],
    }

    class FakeClient:
        def set_tag(self, run_id: str, key: str, value: str) -> None:
            calls["run_tags"].append((run_id, key, value))

        def log_param(self, run_id: str, key: str, value: str) -> None:
            calls["run_tags"].append((run_id, key, value))

        def log_metric(self, run_id: str, key: str, value: float) -> None:
            calls["metrics"].append((run_id, key, value))

        def log_artifact(self, run_id: str, local_path: str, artifact_path: str | None = None) -> None:
            calls["run_tags"].append((run_id, Path(local_path).name, artifact_path or ""))

        def set_model_version_tag(self, name: str, version: str, key: str, value: str) -> None:
            calls["version_tags"].append((name, version, key, value))

    class FakeTracking:
        MlflowClient = FakeClient

    class FakeMlflow:
        tracking = FakeTracking()

    monkeypatch.setitem(sys.modules, "mlflow", FakeMlflow())

    runner.tag_mlflow_promotion_evidence(
        {
            "mlflow_run_id": "run-002",
            "gate_decision": "rejected",
            "gate_reason": "candidate_regressed_on_reference_panel",
            "promotion_status": "rejected_reference_regression",
            "gate_eval_profile": "fast",
            "selected_metric": "pixel_aupimo_1e-5_1e-3",
            "active_metric_value": 0.0,
            "candidate_metric_value": 0.0,
            "metric_delta": 0.0,
            "active_false_negatives": 7,
            "candidate_false_negatives": 7,
            "fn_delta": 0,
            "active_good_red_count": 1,
            "candidate_good_red_count": 1,
            "good_red_delta": 0,
            "localization_gate": {"passed": False, "delta": 0.0},
            "classification_gate": {
                "active_false_negatives": 7,
                "candidate_false_negatives": 7,
                "fn_delta": 0,
                "active_good_red_count": 1,
                "candidate_good_red_count": 1,
                "good_red_delta": 0,
                "passed": True,
            },
            "classification_progress": {
                "improved": False,
                "non_regression": True,
                "summary": "stable: FN 7 -> 7",
            },
        }
    )

    assert ("run-002", "gate_decision", "rejected") in calls["run_tags"]
    assert ("run-002", "gate.reason", "candidate_regressed_on_reference_panel") in calls["run_tags"]
    assert ("run-002", "gate.passed", 0.0) in calls["metrics"]
    assert ("run-002", "gate.active_false_negatives", 7.0) in calls["metrics"]
    assert ("run-002", "gate.candidate_good_red_count", 1.0) in calls["metrics"]
    assert ("run-002", "gate.localization.passed", 0.0) in calls["metrics"]
    assert ("run-002", "gate.classification.passed", 1.0) in calls["metrics"]
    assert ("run-002", "gate.classification_progress.non_regression", 1.0) in calls["metrics"]
    assert calls["version_tags"] == []


def test_drift_cycle_triggers_on_confirmed_drift(tmp_path: Path, monkeypatch) -> None:
    plan = tmp_path / "drift.csv"
    _write_replay(plan, scenario_id=runner.DRIFT_SCENARIO_ID, rows=1)
    monkeypatch.setattr(runner, "REPLAY_PLANS", {runner.DRIFT_SCENARIO_ID: plan})
    monkeypatch.setattr(runner, "ACTIVE_REPLAY_SCENARIOS", tmp_path / "missing_replay_scenarios.csv")
    _mock_runtime(monkeypatch)

    summary = runner.run_cycle(_args(tmp_path, scenario_id=runner.DRIFT_SCENARIO_ID, mode="decision-only"))

    assert summary["trigger_lifecycle"] is True
    assert summary["trigger_reason"] == "drift_confirmed"
    assert summary["candidate_dataset_version"] == "feature_ae_good_mvp_v001"


def test_runtime_thresholds_require_reference_calibration(monkeypatch) -> None:
    def _missing_thresholds(*args, **kwargs):
        raise ValueError("missing calibrated decision_thresholds")

    monkeypatch.setattr(runner, "load_feature_ae_decision_thresholds", _missing_thresholds)

    with pytest.raises(ValueError, match="missing calibrated decision_thresholds"):
        runner.resolve_runtime_thresholds("missing_thresholds")


def test_metric_cache_roundtrip(tmp_path: Path) -> None:
    cache_root = tmp_path / "metric_cache"
    output_dir = tmp_path / "cycle_eval"
    output_dir.mkdir()
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "metrics": {"pixel_aupimo_1e-5_1e-3": 0.2, "pixel_ap": 0.1},
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "params.json").write_text(
        json.dumps({"duration_seconds": 1.2}),
        encoding="utf-8",
    )

    runner.store_metric_cache(cache_root, "abc123", output_dir)
    loaded = runner.load_metric_cache(cache_root, "abc123", tmp_path / "loaded_eval")

    assert loaded is not None
    assert loaded["metrics"]["pixel_aupimo_1e-5_1e-3"] == 0.2
    assert (cache_root / "index.json").exists()
    assert (tmp_path / "loaded_eval" / "metrics.json").exists()


def test_metric_cache_key_changes_with_gate_eval_profile(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    manifest = tmp_path / "validation.csv"
    manifest.write_text("image_id,relative_path,is_defective\n", encoding="utf-8")
    masks = tmp_path / "masks.csv"
    masks.write_text("image_id,gt_mask_path\n", encoding="utf-8")

    fast_key = runner.metric_cache_key(
        model_version="model-v1",
        checkpoint_path=checkpoint,
        evaluation_set_path=manifest,
        evaluation_set_id="validation",
        gt_masks_manifest=masks,
        gate_eval_profile="fast",
        threshold_orange=0.02,
        threshold_red=0.05,
    )
    full_key = runner.metric_cache_key(
        model_version="model-v1",
        checkpoint_path=checkpoint,
        evaluation_set_path=manifest,
        evaluation_set_id="validation",
        gt_masks_manifest=masks,
        gate_eval_profile="full",
        threshold_orange=0.02,
        threshold_red=0.05,
    )

    assert fast_key != full_key
