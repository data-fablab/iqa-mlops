from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import run_replay_lifecycle_cycle as runner
from iqa.storage.object_store import InMemoryObjectStore


def _write_fake_checkpoint(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": {"weight": torch.zeros(256)}}, path)


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
        model_cache_root=tmp_path / "models",
        device="cpu",
        batch_size=1,
        epochs=1,
        max_steps=1,
        gate_eval_profile="fast",
        promotion_min_delta=0.0,
        dual_promotion=False,
        localization_promotion_min_delta=0.0,
        classification_require_fn_improvement=True,
        classification_min_image_recall_delta=0.0,
        classification_min_image_ap_delta=0.0,
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
        _write_fake_checkpoint(config.output_checkpoint)
        _write_fake_checkpoint(run_dir / "checkpoint_best_localization.pt")
        _write_fake_checkpoint(run_dir / "checkpoint_best_image.pt")
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
    model_cache_root = tmp_path / "models"
    args = _args(tmp_path, scenario_id=runner.NATURAL_SCENARIO_ID, mode="progressive-train")
    args.model_cache_root = model_cache_root
    args.lifecycle_run_id = "run-a"
    run_dir = model_cache_root / runner.NATURAL_SCENARIO_ID / "run-a" / candidate_version
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

    result = runner.train_progressive_candidate(args, candidate_version, manifest, "feature_ae_natural_cycle_001")

    assert result["run_id"] == "run-001"
    metric_best = json.loads((run_dir / "metric_eval_best.json").read_text(encoding="utf-8"))
    assert metric_best["pixel_aupimo_1e-5_1e-3"]["epoch"] == 1


def test_progressive_candidate_run_dir_is_scoped_by_lifecycle_run_id(tmp_path: Path) -> None:
    args_a = _args(tmp_path, scenario_id=runner.NATURAL_SCENARIO_ID, mode="progressive-train")
    args_b = _args(tmp_path, scenario_id=runner.NATURAL_SCENARIO_ID, mode="progressive-train")
    args_a.model_cache_root = tmp_path / "models"
    args_b.model_cache_root = tmp_path / "models"
    args_a.lifecycle_run_id = "run-a"
    args_b.lifecycle_run_id = "run-b"
    candidate_version = "rd_feature_ae_gated_natural_cycle_001"

    run_dir_a = runner.progressive_candidate_run_dir(args_a, candidate_version)
    run_dir_b = runner.progressive_candidate_run_dir(args_b, candidate_version)

    assert run_dir_a != run_dir_b
    assert run_dir_a == tmp_path / "models" / runner.NATURAL_SCENARIO_ID / "run-a" / candidate_version
    assert run_dir_b == tmp_path / "models" / runner.NATURAL_SCENARIO_ID / "run-b" / candidate_version


def test_feature_ae_checkpoint_validation_rejects_corrupt_checkpoint(tmp_path: Path) -> None:
    corrupt = tmp_path / "checkpoint_best_image.pt"
    corrupt.write_bytes(b"classification")

    with pytest.raises(ValueError, match="too small"):
        runner.validate_feature_ae_checkpoint_file(corrupt, role="classification")


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


def test_lifecycle_asset_preflight_accepts_materialized_images_and_masks(tmp_path: Path, monkeypatch) -> None:
    image_root = tmp_path / "source_datasets" / "hss-iad"
    image = image_root / "Casting_class1" / "test" / "defective" / "part.jpg"
    mask = image_root / "Casting_class1" / "ground_truth" / "defective" / "part_mask.png"
    image.parent.mkdir(parents=True)
    mask.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    mask.write_bytes(b"mask")
    plan = tmp_path / "replay.csv"
    plan.write_text(
        "scenario_id,relative_paths,gt_mask_paths\n"
        f"{runner.PIECE_B_TO_PIECE_A_P4_DRIFT_SCENARIO_ID},Casting_class1/test/defective/part.jpg,Casting_class1/ground_truth/defective/part_mask.png\n",
        encoding="utf-8",
    )
    reference_eval = tmp_path / "reference_eval.csv"
    reference_eval.write_text(
        "relative_paths,gt_mask_paths\n"
        "Casting_class1/test/defective/part.jpg,Casting_class1/ground_truth/defective/part_mask.png\n",
        encoding="utf-8",
    )
    reference_gt = tmp_path / "reference_gt.csv"
    reference_gt.write_text(
        "image_id,gt_mask_path\n"
        "img_001,Casting_class1/ground_truth/defective/part_mask.png\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "REPLAY_PLANS", {runner.PIECE_B_TO_PIECE_A_P4_DRIFT_SCENARIO_ID: plan})
    monkeypatch.setattr(runner, "ACTIVE_REPLAY_SCENARIOS", tmp_path / "missing_replay_scenarios.csv")
    args = _args(tmp_path, scenario_id=runner.PIECE_B_TO_PIECE_A_P4_DRIFT_SCENARIO_ID, mode="progressive-train")
    args.image_root = image_root
    args.reference_eval_manifest = reference_eval
    args.reference_gt_masks_manifest = reference_gt

    runner.preflight_lifecycle_assets(args, replay_rows=runner.load_replay_rows(args.scenario_id))


def test_lifecycle_asset_preflight_fails_before_training_when_mask_missing(tmp_path: Path, monkeypatch) -> None:
    image_root = tmp_path / "source_datasets" / "hss-iad"
    image = image_root / "Casting_class1" / "test" / "defective" / "part.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    plan = tmp_path / "replay.csv"
    plan.write_text(
        "scenario_id,relative_paths,gt_mask_paths\n"
        f"{runner.PIECE_B_TO_PIECE_A_P4_DRIFT_SCENARIO_ID},Casting_class1/test/defective/part.jpg,Casting_class1/ground_truth/defective/missing_mask.png\n",
        encoding="utf-8",
    )
    reference_eval = tmp_path / "reference_eval.csv"
    reference_eval.write_text("relative_paths,gt_mask_paths\n", encoding="utf-8")
    reference_gt = tmp_path / "reference_gt.csv"
    reference_gt.write_text(
        "image_id,gt_mask_path\n"
        "img_001,Casting_class1/ground_truth/defective/missing_mask.png\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "REPLAY_PLANS", {runner.PIECE_B_TO_PIECE_A_P4_DRIFT_SCENARIO_ID: plan})
    monkeypatch.setattr(runner, "ACTIVE_REPLAY_SCENARIOS", tmp_path / "missing_replay_scenarios.csv")
    args = _args(tmp_path, scenario_id=runner.PIECE_B_TO_PIECE_A_P4_DRIFT_SCENARIO_ID, mode="progressive-train")
    args.image_root = image_root
    args.reference_eval_manifest = reference_eval
    args.reference_gt_masks_manifest = reference_gt

    with pytest.raises(FileNotFoundError, match="reference_gt.csv"):
        runner.preflight_lifecycle_assets(args, replay_rows=runner.load_replay_rows(args.scenario_id))


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


def test_initial_registered_runtimes_seed_role_specific_chains_and_api_events(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan = tmp_path / "natural.csv"
    _write_replay(plan, scenario_id=runner.NATURAL_SCENARIO_ID, rows=10)
    monkeypatch.setattr(runner, "REPLAY_PLANS", {runner.NATURAL_SCENARIO_ID: plan})
    monkeypatch.setattr(runner, "ACTIVE_REPLAY_SCENARIOS", tmp_path / "missing_replay_scenarios.csv")
    _mock_runtime(monkeypatch)
    classifier_checkpoint = tmp_path / "classifier.pt"
    localizer_checkpoint = tmp_path / "localizer.pt"
    _write_fake_checkpoint(classifier_checkpoint)
    _write_fake_checkpoint(localizer_checkpoint)
    thresholds = {"threshold_orange": 0.42, "threshold_red": 0.84, "threshold_source": "test"}
    runtimes = {
        "classification": runner.ActiveRuntimeModel(
            version="rd_feature_ae_gated_natural_cycle_001",
            checkpoint=classifier_checkpoint,
            decision_thresholds=thresholds,
            registry_model_name="feature_ae_classifier__production_replay_natural",
            registry_stage="test",
            registry_alias="test",
            registered_model_version="2",
            registry_status="loaded_from_registry_initial",
            registry_source_of_truth="mlflow_registry",
        ),
        "localization": runner.ActiveRuntimeModel(
            version="rd_feature_ae_gated_natural_cycle_005",
            checkpoint=localizer_checkpoint,
            decision_thresholds=thresholds,
            registry_model_name="feature_ae_localization__production_replay_natural",
            registry_stage="test",
            registry_alias="test",
            registered_model_version="5",
            registry_status="loaded_from_registry_initial",
            registry_source_of_truth="mlflow_registry",
        ),
    }
    monkeypatch.setattr(
        runner,
        "resolve_registered_initial_runtime",
        lambda args, *, model_name, role, fallback_thresholds: runtimes[role],
    )
    api_events: list[dict[str, object]] = []

    def fake_emit_lifecycle_api_event(args, event_type, *, metrics=None, **payload):
        api_events.append(runner._lifecycle_api_payload(args, event_type, metrics=metrics, **payload))

    monkeypatch.setattr(runner, "emit_lifecycle_api_event", fake_emit_lifecycle_api_event)
    args = _args(tmp_path, scenario_id=runner.NATURAL_SCENARIO_ID, mode="progressive-decision")
    args.initial_classification_registered_model = "feature_ae_classifier__production_replay_natural"
    args.initial_localization_registered_model = "feature_ae_localization__production_replay_natural"

    summary = runner.run_cycle(args)

    assert summary["active_model_initial"] == "rd_feature_ae_gated_natural_cycle_001"
    assert summary["active_localization_model_initial"] == "rd_feature_ae_gated_natural_cycle_005"
    assert summary["promotion_chain"] == ["rd_feature_ae_gated_natural_cycle_001"]
    assert summary["classification_promotion_chain"] == ["rd_feature_ae_gated_natural_cycle_001"]
    assert summary["localization_promotion_chain"] == ["rd_feature_ae_gated_natural_cycle_005"]
    assert summary["active_classification_runtime_final"]["version"] == "rd_feature_ae_gated_natural_cycle_001"
    assert summary["active_localization_runtime_final"]["version"] == "rd_feature_ae_gated_natural_cycle_005"
    run_started = next(event for event in api_events if event["event_type"] == "run_started")
    assert run_started["active_classification_model_version"] == "rd_feature_ae_gated_natural_cycle_001"
    assert run_started["active_localization_model_version"] == "rd_feature_ae_gated_natural_cycle_005"


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
    assert train_calls[0].metric_eval_manifest_path == args.reference_eval_manifest
    assert train_calls[0].gt_masks_manifest == args.reference_gt_masks_manifest
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
        _write_fake_checkpoint(config.output_checkpoint)
        _write_fake_checkpoint(run_dir / "checkpoint_best_localization.pt")
        _write_fake_checkpoint(run_dir / "checkpoint_best_image.pt")
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


def test_classification_selection_prefers_lower_false_negatives_before_ap(tmp_path: Path, monkeypatch) -> None:
    args = _args(tmp_path, scenario_id=runner.PIECE_B_TO_PIECE_A_P4_DRIFT_SCENARIO_ID, mode="progressive-train")
    args.lifecycle_run_id = "selection_run"
    selection_manifest = tmp_path / "classification_selection.csv"
    selection_manifest.write_text(
        "image_id,relative_path,label,is_defective\n"
        "sel_good,Casting_class1/train/good/sel_good.jpg,good,false\n"
        "sel_def,Casting_class1/ground_truth/defective/sel_def.jpg,defective,true\n",
        encoding="utf-8",
    )
    args.classification_selection_manifest = selection_manifest
    candidate_version = "rd_feature_ae_gated_natural_cycle_001"
    run_dir = runner.progressive_candidate_run_dir(args, candidate_version)
    checkpoint_001 = run_dir / "checkpoint_epoch_001.pt"
    checkpoint_002 = run_dir / "checkpoint_epoch_002.pt"
    _write_fake_checkpoint(checkpoint_001)
    _write_fake_checkpoint(checkpoint_002)
    _write_fake_checkpoint(run_dir / "checkpoint_best_image.pt")

    def fake_evaluate_model_pair_on_panel(*_args, candidate_checkpoint_path: Path, **_kwargs) -> dict:
        if candidate_checkpoint_path.name == "checkpoint_epoch_002.pt":
            candidate_metrics = {
                "false_negatives": 0,
                "image_recall": 1.0,
                "image_ap": 0.60,
                "good_red_count": 0,
            }
        else:
            candidate_metrics = {
                "false_negatives": 1,
                "image_recall": 0.5,
                "image_ap": 0.99,
                "good_red_count": 0,
            }
        return {
            "active_metrics_on_eval_set": {"good_red_count": 0},
            "candidate_metrics_on_eval_set": candidate_metrics,
            "candidate_eval_metrics_path": str(tmp_path / f"{candidate_checkpoint_path.stem}.json"),
        }

    monkeypatch.setattr(runner, "evaluate_model_pair_on_panel", fake_evaluate_model_pair_on_panel)
    active_runtime = runner.ActiveRuntimeModel(
        version="active",
        checkpoint=tmp_path / "active.pt",
        decision_thresholds={"threshold_orange": 0.1, "threshold_red": 0.2},
        registry_model_name="feature_ae_classifier__stable",
        registry_stage="test",
    )
    artifacts = runner.LifecycleArtifacts(
        events_path=tmp_path / "events.jsonl",
        lots_path=tmp_path / "lots.jsonl",
        cycles_path=tmp_path / "cycles.jsonl",
        summary_path=tmp_path / "summary.json",
        progress_path=tmp_path / "progress.json",
        lifecycle_events_path=tmp_path / "lifecycle_events.jsonl",
        timings_path=tmp_path / "timings.jsonl",
    )
    state = runner.CycleState(
        scenario_id=args.scenario_id,
        mode="progressive-train",
        run_id=args.lifecycle_run_id,
        output_dir=tmp_path / "out",
    )

    selection = runner.select_classification_candidate_checkpoint(
        args,
        cycle_dir=tmp_path / "cycle_001",
        candidate_version=candidate_version,
        default_checkpoint=run_dir / "checkpoint_best_image.pt",
        active_runtime=active_runtime,
        artifacts=artifacts,
        state=state,
    )

    assert selection["reason"] == "selected_by_false_negatives_recall_ap"
    assert Path(selection["selected_checkpoint"]).name == "checkpoint_epoch_002.pt"
    assert selection["selected_false_negatives"] == 0


def _run_dual_promotion_case(
    tmp_path: Path,
    monkeypatch,
    *,
    active_pixel: float,
    candidate_pixel: float,
    active_fn: int,
    candidate_fn: int,
    active_good_red: int = 0,
    candidate_good_red: int = 0,
) -> tuple[dict[str, object], dict[str, object], list[dict[str, str]]]:
    plan = tmp_path / "natural.csv"
    _write_replay(plan, scenario_id=runner.NATURAL_SCENARIO_ID, rows=60)
    monkeypatch.setattr(runner, "REPLAY_PLANS", {runner.NATURAL_SCENARIO_ID: plan})
    monkeypatch.setattr(runner, "ACTIVE_REPLAY_SCENARIOS", tmp_path / "missing_replay_scenarios.csv")
    _mock_runtime(monkeypatch)
    monkeypatch.setattr(runner, "apply_decision_thresholds_to_evaluation", lambda evaluation, decision_thresholds: evaluation)
    monkeypatch.setattr(
        runner,
        "thresholds_from_evaluation_scores",
        lambda *args, **kwargs: {
            "method": "test",
            "threshold_orange": 0.5,
            "threshold_red": 0.9,
            "threshold_source": "test",
        },
    )
    registry_calls: list[dict[str, str]] = []

    def fake_register(**kwargs):
        registry_calls.append(kwargs)
        return {
            "registered_model_name": runner.registered_model_name(kwargs["scenario_id"], base_name=kwargs.get("model_name_base", "feature_ae")),
            "version": str(len(registry_calls)),
            "stage": kwargs["stage"],
            "alias": kwargs["stage"],
            "source_of_truth": "mlflow_registry",
        }

    monkeypatch.setattr(runner, "register_run_to_model", fake_register)
    monkeypatch.setattr(runner, "tag_mlflow_promotion_evidence", lambda cycle: None)

    def fake_evaluate(args, model_version, checkpoint_path, evaluation_set_path, output_dir, **kwargs):
        is_candidate = model_version.endswith("cycle_001")
        is_localization = "reference_localization" in {part.lower() for part in output_dir.parts}
        pixel = candidate_pixel if is_candidate and is_localization else active_pixel if is_localization else 0.5
        false_negatives = candidate_fn if is_candidate and not is_localization else active_fn if not is_localization else 0
        good_red_count = candidate_good_red if is_candidate and not is_localization else active_good_red if not is_localization else 0
        image_recall = 1.0 - (false_negatives * 0.1)
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics = {
            "pixel_aupimo_1e-5_1e-3": pixel,
            "pixel_ap": pixel / 4,
            "image_ap": 0.5 + image_recall / 10,
            "image_recall": image_recall,
            "false_negatives": false_negatives,
            "good_red_count": good_red_count,
            "good_red_rate": float(good_red_count) / 10.0,
        }
        (output_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "metrics": metrics,
                    "images": [
                        {"image_id": f"good_{index:03d}", "score": 0.1 + index * 0.001, "is_defective": False}
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
    args.dual_promotion = True
    args.max_cycles = 1
    args.max_good_red_regression = 1

    summary = runner.run_cycle(args)
    cycle = json.loads((Path(summary["output_dir"]) / "cycles.jsonl").read_text(encoding="utf-8"))
    return summary, cycle, registry_calls


def test_dual_promotion_promotes_localization_only_when_pixel_improves_without_fn_gain(tmp_path: Path, monkeypatch) -> None:
    summary, cycle, registry_calls = _run_dual_promotion_case(
        tmp_path,
        monkeypatch,
        active_pixel=0.3,
        candidate_pixel=0.5,
        active_fn=2,
        candidate_fn=2,
    )

    assert cycle["localization_promotion_status"] == "promoted"
    assert cycle["classification_promotion_status"] == "rejected_no_classification_improvement"
    assert cycle["gate_decision"] == "partially_passed"
    assert cycle["promotion_status"] == "partially_promoted"
    assert cycle["dual_promotion_outcome"] == "localization_promoted__classification_rejected_no_classification_improvement"
    assert cycle["activated_for_next_events"] is True
    assert cycle["decision_model_activated_for_next_events"] is False
    assert cycle["localization_activated_for_next_events"] is True
    assert cycle["classification_activated_for_next_events"] is False
    assert "rd_feature_ae_gated_natural_cycle_001" in summary["models_promoted"]
    assert "rd_feature_ae_gated_natural_cycle_001" not in summary["rejected_candidates"]
    assert summary["localization_promotion_chain"] == [
        runner.DEFAULT_FEATURE_AE_MODEL_VERSION,
        "rd_feature_ae_gated_natural_cycle_001",
    ]
    assert summary["classification_promotion_chain"] == [runner.DEFAULT_FEATURE_AE_MODEL_VERSION]
    assert [call["model_name_base"] for call in registry_calls] == [runner.LOCALIZATION_MODEL_NAME_BASE]


def test_dual_promotion_promotes_classification_only_when_fn_decreases(tmp_path: Path, monkeypatch) -> None:
    summary, cycle, registry_calls = _run_dual_promotion_case(
        tmp_path,
        monkeypatch,
        active_pixel=0.7,
        candidate_pixel=0.6,
        active_fn=2,
        candidate_fn=1,
    )

    assert cycle["localization_promotion_status"] == "rejected_reference_regression"
    assert cycle["classification_promotion_status"] == "promoted"
    assert cycle["gate_decision"] == "passed"
    assert cycle["promotion_status"] == "promoted"
    assert cycle["dual_promotion_outcome"] == "localization_rejected_reference_regression__classification_promoted"
    assert cycle["activated_for_next_events"] is True
    assert cycle["decision_model_activated_for_next_events"] is True
    assert cycle["localization_activated_for_next_events"] is False
    assert cycle["classification_activated_for_next_events"] is True
    assert summary["localization_promotion_chain"] == [runner.DEFAULT_FEATURE_AE_MODEL_VERSION]
    assert summary["classification_promotion_chain"] == [
        runner.DEFAULT_FEATURE_AE_MODEL_VERSION,
        "rd_feature_ae_gated_natural_cycle_001",
    ]
    assert [call["model_name_base"] for call in registry_calls] == [runner.CLASSIFICATION_MODEL_NAME_BASE]


def test_dual_promotion_promotes_both_roles_to_distinct_registered_models(tmp_path: Path, monkeypatch) -> None:
    summary, cycle, registry_calls = _run_dual_promotion_case(
        tmp_path,
        monkeypatch,
        active_pixel=0.3,
        candidate_pixel=0.6,
        active_fn=2,
        candidate_fn=1,
    )

    assert cycle["localization_promotion_status"] == "promoted"
    assert cycle["classification_promotion_status"] == "promoted"
    assert cycle["gate_decision"] == "passed"
    assert cycle["promotion_status"] == "promoted"
    assert cycle["dual_promotion_outcome"] == "localization_promoted__classification_promoted"
    assert summary["localization_promotion_chain"][-1] == "rd_feature_ae_gated_natural_cycle_001"
    assert summary["classification_promotion_chain"][-1] == "rd_feature_ae_gated_natural_cycle_001"
    assert [call["model_name_base"] for call in registry_calls] == [
        runner.LOCALIZATION_MODEL_NAME_BASE,
        runner.CLASSIFICATION_MODEL_NAME_BASE,
    ]
    assert [call["model_artifact_path"] for call in registry_calls] == [
        "model_localization",
        "model_classification",
    ]
    assert cycle["localization_registered_model_name"] == "feature_ae_localization__production_replay_natural"
    assert cycle["classification_registered_model_name"] == "feature_ae_classifier__production_replay_natural"
    assert cycle["localization_checkpoint_sha256"]
    assert cycle["classification_checkpoint_sha256"]


def test_piece_a_p4_demo_gate_promotes_on_p4_panel_and_reports_piece_b_regression(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan = tmp_path / "piece_a_p4.csv"
    lines = [
        "event_id,piece_event_id,scenario_id,lot_id,source_class,dataset_version,scenario_phase,relative_paths,image_ids,is_defective,has_mask"
    ]
    for index in range(60):
        lines.append(
            f"event_{index:03d},piece_{index:03d},{runner.PIECE_B_TO_PIECE_A_P4_DRIFT_SCENARIO_ID},LOT-001,"
            f"Casting_class1,{runner.PIECE_B_TO_PIECE_A_P4_DRIFT_SCENARIO_ID}_v001,correction_replay,"
            f"Casting_class1/train/good/part_{index:03d}.jpg,img_{index:03d},false,false"
        )
    plan.write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(runner, "REPLAY_PLANS", {runner.PIECE_B_TO_PIECE_A_P4_DRIFT_SCENARIO_ID: plan})
    monkeypatch.setattr(runner, "ACTIVE_REPLAY_SCENARIOS", tmp_path / "missing_replay_scenarios.csv")
    _mock_runtime(monkeypatch)
    monkeypatch.setattr(runner, "apply_decision_thresholds_to_evaluation", lambda evaluation, decision_thresholds: evaluation)
    monkeypatch.setattr(
        runner,
        "thresholds_from_evaluation_scores",
        lambda *args, **kwargs: {
            "method": "test",
            "threshold_orange": 0.5,
            "threshold_red": 0.9,
            "threshold_source": "test",
        },
    )
    registry_calls: list[dict[str, str]] = []

    def fake_register(**kwargs):
        registry_calls.append(kwargs)
        return {
            "registered_model_name": runner.registered_model_name(
                kwargs["scenario_id"],
                base_name=kwargs.get("model_name_base", "feature_ae"),
            ),
            "version": str(len(registry_calls)),
            "stage": kwargs["stage"],
            "alias": kwargs["stage"],
            "source_of_truth": "mlflow_registry",
        }

    monkeypatch.setattr(runner, "register_run_to_model", fake_register)
    monkeypatch.setattr(runner, "tag_mlflow_promotion_evidence", lambda cycle: None)

    def fake_evaluate(args, model_version, checkpoint_path, evaluation_set_path, output_dir, **kwargs):
        del args, checkpoint_path, evaluation_set_path, kwargs
        is_candidate = model_version.endswith("cycle_001")
        panel_name = output_dir.parent.name
        output_dir.mkdir(parents=True, exist_ok=True)
        if panel_name == "piece_a_p4_correction_localization":
            pixel = 0.50 if is_candidate else 0.20
            metrics = {
                "pixel_aupimo_1e-5_1e-3": pixel,
                "pixel_ap": pixel / 2,
                "false_negatives": 0,
                "image_recall": 1.0,
                "image_ap": 0.5,
                "good_red_count": 0,
            }
        elif panel_name in {"piece_a_p4_correction_classification", "classification_selection_checkpoint_best_image"}:
            false_negatives = 6 if is_candidate else 5
            metrics = {
                "pixel_aupimo_1e-5_1e-3": 0.3,
                "pixel_ap": 0.1,
                "false_negatives": false_negatives,
                "image_recall": 0.70 if is_candidate else 0.75,
                "image_ap": 0.90 if is_candidate else 0.40,
                "good_red_count": 0,
            }
        elif panel_name == "report_piece_b_non_regression_localization":
            pixel = 0.30 if is_candidate else 0.70
            metrics = {
                "pixel_aupimo_1e-5_1e-3": pixel,
                "pixel_ap": pixel / 2,
                "false_negatives": 0,
                "image_recall": 1.0,
                "image_ap": 0.5,
                "good_red_count": 0,
            }
        elif panel_name == "report_piece_b_non_regression_classification":
            false_negatives = 4 if is_candidate else 1
            metrics = {
                "pixel_aupimo_1e-5_1e-3": 0.3,
                "pixel_ap": 0.1,
                "false_negatives": false_negatives,
                "image_recall": 0.60 if is_candidate else 0.90,
                "image_ap": 0.40 if is_candidate else 0.80,
                "good_red_count": 0,
            }
        else:
            metrics = {
                "pixel_aupimo_1e-5_1e-3": 0.3,
                "pixel_ap": 0.1,
                "false_negatives": 0,
                "image_recall": 1.0,
                "image_ap": 0.5,
                "good_red_count": 0,
            }
        (output_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "metrics": metrics,
                    "images": [
                        {"image_id": f"image_{index:03d}", "score": 0.1 + index * 0.001, "is_defective": False}
                        for index in range(6)
                    ],
                }
            ),
            encoding="utf-8",
        )
        return {
            "model_version": model_version,
            "checkpoint_path": str(output_dir / "checkpoint.pt"),
            "metrics": metrics,
            "metrics_path": str(output_dir / "metrics.json"),
        }

    monkeypatch.setattr(runner, "evaluate_reference_model_on_set", fake_evaluate)
    selection_manifest = tmp_path / "classification_selection_piece_a_p4.csv"
    selection_manifest.write_text(
        "image_id,relative_path,label,is_defective\n"
        "p4_good,Casting_class1/train/good/p4_good_2_3.jpg,good,false\n"
        "p4_def,Casting_class1/test/defective/p4_def_2_3.jpg,defective,true\n",
        encoding="utf-8",
    )
    args = _args(tmp_path, scenario_id=runner.PIECE_B_TO_PIECE_A_P4_DRIFT_SCENARIO_ID, mode="progressive-train")
    args.dual_promotion = True
    args.max_cycles = 1
    args.external_drift_confirmed = True
    args.classification_selection_manifest = selection_manifest
    for index in range(60):
        replay_image = args.image_root / "Casting_class1" / "train" / "good" / f"part_{index:03d}.jpg"
        replay_image.parent.mkdir(parents=True, exist_ok=True)
        replay_image.write_bytes(b"image")
    reference_image = args.image_root / "Casting_class1" / "test" / "good" / "ref_001.jpg"
    reference_image.parent.mkdir(parents=True, exist_ok=True)
    reference_image.write_bytes(b"image")

    summary = runner.run_cycle(args)
    cycle = json.loads((Path(summary["output_dir"]) / "cycles.jsonl").read_text(encoding="utf-8"))

    assert cycle["promotion_objective"] == runner.PIECE_A_P4_DEMO_PROMOTION_OBJECTIVE
    assert cycle["piece_b_non_regression_policy"] == runner.PIECE_B_NON_REGRESSION_REPORT_ONLY
    assert cycle["gate_evaluation_set_id"] == selection_manifest.stem
    assert cycle["localization_promotion_status"] == "promoted"
    assert cycle["classification_promotion_status"] == "promoted"
    assert cycle["dual_promotion_outcome"] == "localization_promoted__classification_promoted"
    assert cycle["classification_gate"]["selected_metric"] == "image_ap"
    assert cycle["classification_gate"]["fn_delta"] == 1
    assert cycle["classification_gate"]["metric_ok"] is True
    assert cycle["piece_b_non_regression_localization_metric_delta"] < 0
    assert cycle["piece_b_non_regression_classification_metric_delta"] < 0
    assert [call["model_name_base"] for call in registry_calls] == [
        runner.LOCALIZATION_MODEL_NAME_BASE,
        runner.CLASSIFICATION_MODEL_NAME_BASE,
    ]


def test_dual_promotion_good_red_guard_blocks_classifier_not_localizer(tmp_path: Path, monkeypatch) -> None:
    summary, cycle, registry_calls = _run_dual_promotion_case(
        tmp_path,
        monkeypatch,
        active_pixel=0.3,
        candidate_pixel=0.6,
        active_fn=2,
        candidate_fn=1,
        active_good_red=0,
        candidate_good_red=3,
    )

    assert cycle["localization_promotion_status"] == "promoted"
    assert cycle["classification_promotion_status"] == "rejected_operational_guardrail"
    assert cycle["gate_decision"] == "partially_passed"
    assert cycle["promotion_status"] == "partially_promoted"
    assert cycle["dual_promotion_outcome"] == "localization_promoted__classification_rejected_operational_guardrail"
    assert summary["localization_promotion_chain"][-1] == "rd_feature_ae_gated_natural_cycle_001"
    assert summary["classification_promotion_chain"] == [runner.DEFAULT_FEATURE_AE_MODEL_VERSION]
    assert [call["model_name_base"] for call in registry_calls] == [runner.LOCALIZATION_MODEL_NAME_BASE]


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


def test_runner_pushes_lifecycle_epoch_event_best_effort(tmp_path: Path, monkeypatch) -> None:
    args = _args(tmp_path, scenario_id=runner.NATURAL_SCENARIO_ID, mode="progressive-train")
    args.lifecycle_run_id = "replay_lifecycle_001"
    calls: list[dict[str, object]] = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

        def read(self) -> bytes:
            return b"{}"

    def fake_urlopen(request, timeout):
        calls.append(
            {
                "url": request.full_url,
                "data": json.loads(request.data.decode("utf-8")),
                "token": request.headers.get("X-iqa-service-token"),
                "timeout": timeout,
            }
        )
        return FakeResponse()

    monkeypatch.setenv("IQA_API_URL", "http://iqa-api:8000")
    monkeypatch.setenv("IQA_SERVICE_TOKEN", "service-secret")
    monkeypatch.setattr(runner.urllib.request, "urlopen", fake_urlopen)

    runner.emit_lifecycle_api_event(
        args,
        "epoch_completed",
        cycle_id="cycle_001",
        epoch=2,
        candidate_version="rd_feature_ae_gated_natural_cycle_001",
        metrics={
            "pixel_aupimo_1e-5_1e-3": 0.12,
            "pixel_ap": 0.34,
            "image_ap": 0.56,
            "image_path": "D:\\MLOPS\\secret.png",
        },
    )

    assert calls == [
        {
            "url": "http://iqa-api:8000/internal/lifecycle/events",
            "data": {
                "candidate_init_policy": "fresh",
                "candidate_version": "rd_feature_ae_gated_natural_cycle_001",
                "cycle_id": "cycle_001",
                "epoch": 2,
                "event_type": "epoch_completed",
                "lifecycle_run_id": "replay_lifecycle_001",
                "metrics": {
                    "image_ap": 0.56,
                    "pixel_ap": 0.34,
                    "pixel_aupimo_1e-5_1e-3": 0.12,
                },
                "scenario_id": runner.NATURAL_SCENARIO_ID,
            },
            "token": "service-secret",
            "timeout": 2,
        }
    ]


def test_runner_continues_if_api_push_fails(tmp_path: Path, monkeypatch) -> None:
    args = _args(tmp_path, scenario_id=runner.NATURAL_SCENARIO_ID, mode="progressive-train")
    args.lifecycle_run_id = "replay_lifecycle_001"
    monkeypatch.setenv("IQA_API_URL", "http://iqa-api:8000")
    monkeypatch.setattr(
        runner.urllib.request,
        "urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("api offline")),
    )

    runner.emit_lifecycle_api_event(args, "run_started")


def test_tag_mlflow_promotion_evidence_tags_run_and_model_version(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, list[tuple[str, ...] | tuple[str, str, float] | tuple[str, str, str | None]]] = {
        "run_tags": [],
        "version_tags": [],
        "descriptions": [],
        "params": [],
        "metrics": [],
        "artifacts": [],
        "tracking_uris": [],
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

        @staticmethod
        def set_tracking_uri(uri: str) -> None:
            calls["tracking_uris"].append((uri,))

    monkeypatch.setitem(sys.modules, "mlflow", FakeMlflow())
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("IQA_MLFLOW_TRACKING_URI", "http://mlflow:5000")

    cycle = {
        "mlflow_run_id": "run-001",
        "registered_model_name": "feature_ae__production_replay_natural",
        "registered_model_version": "4",
        "candidate_version": "rd_feature_ae_gated_natural_cycle_001",
        "gate_decision": "passed",
        "gate_reason": "candidate_improved_on_reference_panel",
        "gate_eval_profile": "fast",
        "candidate_init_policy": "active",
        "candidate_initial_model_version": runner.DEFAULT_FEATURE_AE_MODEL_VERSION,
        "candidate_initial_checkpoint": ".cache/iqa/models/bootstrap/checkpoint.pt",
        "candidate_initial_checkpoint_sha256": "a" * 64,
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
        "localization_gate_reason": "localization_promoted",
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
        "classification_gate_reason": "classification_promoted",
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

    runner.tag_mlflow_promotion_evidence(cycle)

    assert calls["tracking_uris"] == [("http://mlflow:5000",)]
    assert cycle["mlflow_gate_evidence_status"] == "logged"
    assert ("run-001", "gate_decision", "passed") in calls["run_tags"]
    assert ("run-001", "gate_eval_profile", "fast") in calls["run_tags"]
    assert ("run-001", "candidate_init_policy", "active") in calls["run_tags"]
    assert ("run-001", "candidate_initial_checkpoint_sha256", "a" * 64) in calls["run_tags"]
    assert ("run-001", "localization_gate_reason", "localization_promoted") in calls["run_tags"]
    assert ("run-001", "classification_gate_reason", "classification_promoted") in calls["run_tags"]
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
    assert ("run-001", "mlflow_gate_evidence_status", "logged") in calls["run_tags"]
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

        @staticmethod
        def set_tracking_uri(uri: str) -> None:
            pass

    monkeypatch.setitem(sys.modules, "mlflow", FakeMlflow())

    runner.tag_mlflow_promotion_evidence(
        {
            "mlflow_run_id": "run-002",
            "gate_decision": "rejected",
            "gate_reason": "candidate_regressed_on_reference_panel",
            "promotion_status": "rejected_reference_regression",
            "gate_eval_profile": "fast",
            "candidate_init_policy": "fresh",
            "candidate_initial_model_version": "none",
            "candidate_initial_checkpoint": "",
            "candidate_initial_checkpoint_sha256": "",
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
            "localization_gate_reason": "localization_rejected_reference_regression",
            "classification_gate": {
                "active_false_negatives": 7,
                "candidate_false_negatives": 7,
                "fn_delta": 0,
                "active_good_red_count": 1,
                "candidate_good_red_count": 1,
                "good_red_delta": 0,
                "passed": True,
            },
            "classification_gate_reason": "classification_rejected_no_classification_improvement",
            "classification_progress": {
                "improved": False,
                "non_regression": True,
                "summary": "stable: FN 7 -> 7",
            },
        }
    )

    assert ("run-002", "gate_decision", "rejected") in calls["run_tags"]
    assert ("run-002", "gate.reason", "candidate_regressed_on_reference_panel") in calls["run_tags"]
    assert ("run-002", "candidate_init_policy", "fresh") in calls["run_tags"]
    assert ("run-002", "candidate_initial_model_version", "none") in calls["run_tags"]
    assert (
        "run-002",
        "localization_gate_reason",
        "localization_rejected_reference_regression",
    ) in calls["run_tags"]
    assert ("run-002", "gate.passed", 0.0) in calls["metrics"]
    assert ("run-002", "gate.active_false_negatives", 7.0) in calls["metrics"]
    assert ("run-002", "gate.candidate_good_red_count", 1.0) in calls["metrics"]
    assert ("run-002", "gate.localization.passed", 0.0) in calls["metrics"]
    assert ("run-002", "gate.classification.passed", 1.0) in calls["metrics"]
    assert ("run-002", "gate.classification_progress.non_regression", 1.0) in calls["metrics"]
    assert calls["version_tags"] == []


def test_tag_mlflow_promotion_evidence_records_failure(monkeypatch) -> None:
    class FailingClient:
        def set_tag(self, run_id: str, key: str, value: str) -> None:
            raise RuntimeError("tracking store unreachable")

    class FakeTracking:
        MlflowClient = FailingClient

    class FakeMlflow:
        tracking = FakeTracking()

        @staticmethod
        def set_tracking_uri(uri: str) -> None:
            pass

    monkeypatch.setitem(sys.modules, "mlflow", FakeMlflow())
    cycle = {
        "mlflow_run_id": "run-003",
        "gate_decision": "passed",
    }

    runner.tag_mlflow_promotion_evidence(cycle)

    assert cycle["mlflow_gate_evidence_status"] == "failed"
    assert "tracking store unreachable" in cycle["mlflow_gate_evidence_error"]


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


def test_piece_a_p4_lifecycle_does_not_trigger_from_phase_without_external_confirmation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan = tmp_path / "piece_a_p4.csv"
    plan.write_text(
        "event_id,piece_event_id,scenario_id,scenario_phase,lot_id,source_class,dataset_version,relative_paths,image_ids,is_defective,has_mask\n"
        f"event_001,piece_001,{runner.PIECE_B_TO_PIECE_A_P4_DRIFT_SCENARIO_ID},drift_piece_a_p4_confirmed,LOT-001,Casting_class1,"
        f"{runner.PIECE_B_TO_PIECE_A_P4_DRIFT_SCENARIO_ID},Casting_class1/train/good/p4.jpg,img_001,false,false\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "REPLAY_PLANS", {runner.PIECE_B_TO_PIECE_A_P4_DRIFT_SCENARIO_ID: plan})
    monkeypatch.setattr(runner, "ACTIVE_REPLAY_SCENARIOS", tmp_path / "missing_replay_scenarios.csv")
    _mock_runtime(monkeypatch)

    summary = runner.run_cycle(
        _args(tmp_path, scenario_id=runner.PIECE_B_TO_PIECE_A_P4_DRIFT_SCENARIO_ID, mode="decision-only")
    )

    assert summary["trigger_lifecycle"] is False
    lot = json.loads((Path(summary["output_dir"]) / "lots.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert lot["trigger_lifecycle"] is False
    assert lot["trigger_reason"] == "drift_not_confirmed"


def test_piece_a_p4_lifecycle_external_confirmation_waits_for_correction_replay(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan = tmp_path / "piece_a_p4.csv"
    plan.write_text(
        "event_id,piece_event_id,scenario_id,scenario_phase,lot_id,source_class,dataset_version,relative_paths,image_ids,is_defective,has_mask\n"
        f"event_001,piece_001,{runner.PIECE_B_TO_PIECE_A_P4_DRIFT_SCENARIO_ID},drift_piece_a_p4_confirmed,LOT-001,Casting_class1,"
        f"{runner.PIECE_B_TO_PIECE_A_P4_DRIFT_SCENARIO_ID},Casting_class1/train/good/p4_confirmed.jpg,img_001,false,false\n"
        f"event_002,piece_002,{runner.PIECE_B_TO_PIECE_A_P4_DRIFT_SCENARIO_ID},correction_replay,LOT-002,Casting_class1,"
        f"{runner.PIECE_B_TO_PIECE_A_P4_DRIFT_SCENARIO_ID},Casting_class1/train/good/p4_correction.jpg,img_002,false,false\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "REPLAY_PLANS", {runner.PIECE_B_TO_PIECE_A_P4_DRIFT_SCENARIO_ID: plan})
    monkeypatch.setattr(runner, "ACTIVE_REPLAY_SCENARIOS", tmp_path / "missing_replay_scenarios.csv")
    _mock_runtime(monkeypatch)
    args = _args(tmp_path, scenario_id=runner.PIECE_B_TO_PIECE_A_P4_DRIFT_SCENARIO_ID, mode="decision-only")
    args.external_drift_confirmed = True

    summary = runner.run_cycle(args)

    assert summary["trigger_lifecycle"] is True
    assert summary["trigger_reason"] == "drift_piece_a_p4_confirmed"


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
