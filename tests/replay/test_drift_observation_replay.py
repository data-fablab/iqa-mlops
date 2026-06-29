from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from scripts import run_drift_observation_replay as observer
from scripts import run_replay_lifecycle_cycle as lifecycle


def _args(tmp_path: Path, *, thresholds: Path) -> argparse.Namespace:
    return argparse.Namespace(
        scenario_id=observer.SCENARIO_ID,
        image_root=tmp_path / "images",
        output_root=tmp_path / "out",
        model_cache_root=tmp_path / "models",
        device="cpu",
        target_stage="test",
        max_events=None,
        window_size=2,
        thresholds_config=thresholds,
        api_url="",
        service_token="",
        require_mlflow_registry=False,
        initial_classification_registered_model="feature_ae_classifier__production_replay_natural_piece_b_full",
        initial_localization_registered_model="feature_ae_localization__production_replay_natural_piece_b_full",
    )


def _thresholds(tmp_path: Path) -> Path:
    path = tmp_path / "monitoring_thresholds.yaml"
    path.write_text(
        "\n".join(
            [
                "drift:",
                "  min_window_events: 2",
                "  confirm_windows: 2",
                "  domain_ratio_critical: 0.50",
                "  alert_rate_critical: 0.50",
                "  red_rate_critical: 0.20",
                "  unexpected_red_rate_critical: 0.20",
                "  roi_fail_rate_critical: 0.10",
                "  oracle_fn_rate_critical: 0.05",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _row(index: int, *, phase: str, decision: str, defective: bool = False) -> dict[str, str]:
    return {
        "event_id": f"event_{index:03d}",
        "piece_event_id": f"piece_{index:03d}",
        "lot_id": f"LOT-{index // 2:03d}",
        "scenario_id": observer.SCENARIO_ID,
        "scenario_phase": phase,
        "source_class": "Casting_class1",
        "dataset_version": observer.SCENARIO_ID,
        "relative_paths": f"Casting_class1/train/good/part_{index:03d}.jpg",
        "image_ids": f"img_{index:03d}",
        "is_defective": str(defective).lower(),
        "has_mask": str(defective).lower(),
        "decision": decision,
    }


def _patch_runtime(monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, str]]) -> None:
    monkeypatch.setattr(lifecycle, "load_replay_rows", lambda scenario_id: rows)
    monkeypatch.setattr(lifecycle, "resolve_roi_segmenter_checkpoint", lambda *args, **kwargs: Path("roi.pt"))
    monkeypatch.setattr(lifecycle, "resolve_feature_ae_checkpoint", lambda *args, **kwargs: Path("feature.pt"))
    monkeypatch.setattr(
        lifecycle,
        "resolve_runtime_thresholds",
        lambda *args, **kwargs: {
            "threshold_orange": 0.42,
            "threshold_red": 0.84,
            "threshold_source": "manifest:test",
        },
    )
    monkeypatch.setattr(lifecycle, "resolve_registered_initial_runtime", lambda *args, **kwargs: None)

    def fake_process(row: dict[str, str], **kwargs) -> lifecycle.CycleEvent:
        del kwargs
        return lifecycle.CycleEvent(
            event_id=row["event_id"],
            piece_event_id=row["piece_event_id"],
            lot_id=row["lot_id"],
            scenario_id=row["scenario_id"],
            scenario_phase=row["scenario_phase"],
            source_class=row["source_class"],
            dataset_version=row["dataset_version"],
            relative_path=row["relative_paths"],
            image_path=row["relative_paths"],
            oracle_verdict=lifecycle.oracle_verdict(row),
            decision=row["decision"],
            score=0.1,
            roi_quality_status="ok",
            roi_ratio=0.5,
            threshold_orange=0.42,
            threshold_red=0.84,
            threshold_source="manifest:test",
            roi_mask_path="",
            roi_mask_uri=None,
            roi_probability_path="",
            heatmap_path="",
            heatmap_uri=None,
        )

    monkeypatch.setattr(lifecycle, "process_replay_event", fake_process)


def test_drift_observation_confirms_only_after_observed_degradation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        _row(1, phase="stable_baseline_piece_b", decision="green"),
        _row(2, phase="stable_baseline_piece_b", decision="green"),
        _row(3, phase="drift_piece_a_p4_suspected", decision="red"),
        _row(4, phase="drift_piece_a_p4_suspected", decision="red"),
        _row(5, phase="drift_piece_a_p4_confirmed", decision="red"),
        _row(6, phase="drift_piece_a_p4_confirmed", decision="red"),
    ]
    _patch_runtime(monkeypatch, rows)

    summary = observer.run_observation(_args(tmp_path, thresholds=_thresholds(tmp_path)))

    assert summary["trigger_lifecycle"] is True
    assert summary["trigger_reason"] == "drift_piece_a_p4_confirmed"
    assert summary["ever_suspected"] is True
    assert summary["ever_confirmed"] is True
    assert summary["first_confirmed_window_index"] == 3
    assert summary["last_complete_window_status"] == "confirmed"
    windows = [
        json.loads(line)
        for line in Path(str(summary["windows_path"])).read_text(encoding="utf-8").splitlines()
    ]
    assert [window["status"] for window in windows] == ["clear", "suspected", "confirmed"]
    assert windows[1]["degradation_signals"]["unexpected_red_rate"] is True
    assert windows[0]["metrics"]["drift_score"] == 0.0
    assert windows[1]["metrics"]["drift_score"] == 1.0


def test_drift_observation_does_not_confirm_domain_only_p4(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        _row(1, phase="stable_baseline_piece_b", decision="green"),
        _row(2, phase="stable_baseline_piece_b", decision="green"),
        _row(3, phase="drift_piece_a_p4_suspected", decision="green"),
        _row(4, phase="drift_piece_a_p4_suspected", decision="green"),
        _row(5, phase="drift_piece_a_p4_confirmed", decision="green"),
        _row(6, phase="drift_piece_a_p4_confirmed", decision="green"),
    ]
    _patch_runtime(monkeypatch, rows)

    summary = observer.run_observation(_args(tmp_path, thresholds=_thresholds(tmp_path)))

    assert summary["trigger_lifecycle"] is False
    assert summary["drift_confirmed"] is False
    windows = [
        json.loads(line)
        for line in Path(str(summary["windows_path"])).read_text(encoding="utf-8").splitlines()
    ]
    assert [window["status"] for window in windows] == ["clear", "suspected", "suspected"]
    assert windows[-1]["signals"]["domain_ratio"] is True
    assert windows[-1]["critical_window"] is False


def test_drift_observation_confirms_p4_red_rate_even_when_unexpected_red_is_low(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        _row(1, phase="stable_baseline_piece_b", decision="green"),
        _row(2, phase="stable_baseline_piece_b", decision="green"),
        _row(3, phase="drift_piece_a_p4_suspected", decision="red", defective=True),
        _row(4, phase="drift_piece_a_p4_suspected", decision="red", defective=True),
        _row(5, phase="drift_piece_a_p4_confirmed", decision="red", defective=True),
        _row(6, phase="drift_piece_a_p4_confirmed", decision="red", defective=True),
    ]
    _patch_runtime(monkeypatch, rows)

    summary = observer.run_observation(_args(tmp_path, thresholds=_thresholds(tmp_path)))

    assert summary["trigger_lifecycle"] is True
    assert summary["first_confirmed_window_index"] == 3
    windows = [
        json.loads(line)
        for line in Path(str(summary["windows_path"])).read_text(encoding="utf-8").splitlines()
    ]
    assert [window["status"] for window in windows] == ["clear", "suspected", "confirmed"]
    assert windows[1]["degradation_signals"]["red_rate"] is True
    assert windows[1]["degradation_signals"]["unexpected_red_rate"] is False
    assert windows[1]["critical_window"] is True
    assert windows[1]["metrics"]["drift_score"] == 1.0


def test_drift_observation_does_not_suspect_baseline_degradation_without_p4(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        _row(1, phase="stable_baseline_piece_b", decision="red"),
        _row(2, phase="stable_baseline_piece_b", decision="red"),
        _row(3, phase="stable_baseline_piece_b", decision="red"),
        _row(4, phase="stable_baseline_piece_b", decision="red"),
    ]
    _patch_runtime(monkeypatch, rows)

    summary = observer.run_observation(_args(tmp_path, thresholds=_thresholds(tmp_path)))

    assert summary["trigger_lifecycle"] is False
    assert summary["drift_confirmed"] is False
    windows = [
        json.loads(line)
        for line in Path(str(summary["windows_path"])).read_text(encoding="utf-8").splitlines()
    ]
    assert [window["status"] for window in windows] == ["clear", "clear"]
    assert all(window["signals"]["red_rate"] is True for window in windows)
    assert all(window["signals"]["domain_ratio"] is False for window in windows)


def test_drift_observation_does_not_push_partial_window_to_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        _row(1, phase="stable_baseline_piece_b", decision="green"),
        _row(2, phase="stable_baseline_piece_b", decision="green"),
        _row(3, phase="drift_piece_a_p4_suspected", decision="red"),
    ]
    pushed_windows: list[int] = []
    _patch_runtime(monkeypatch, rows)

    def fake_push(args: argparse.Namespace, drift_evaluation: dict[str, object]) -> dict[str, object]:
        del drift_evaluation
        pushed_windows.append(args.window_events)
        return {"attempted": True, "status": "sent"}

    monkeypatch.setattr(observer, "_push_drift_event", fake_push)

    summary = observer.run_observation(_args(tmp_path, thresholds=_thresholds(tmp_path)))

    windows = [
        json.loads(line)
        for line in Path(str(summary["windows_path"])).read_text(encoding="utf-8").splitlines()
    ]
    assert [window["metrics"]["window_events"] for window in windows] == [2, 1]
    assert [window["window_complete"] for window in windows] == [True, False]
    assert pushed_windows == [2]


def test_drift_observation_partial_final_window_does_not_clear_confirmed_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        _row(1, phase="stable_baseline_piece_b", decision="green"),
        _row(2, phase="stable_baseline_piece_b", decision="green"),
        _row(3, phase="drift_piece_a_p4_suspected", decision="red"),
        _row(4, phase="drift_piece_a_p4_suspected", decision="red"),
        _row(5, phase="drift_piece_a_p4_confirmed", decision="red"),
        _row(6, phase="drift_piece_a_p4_confirmed", decision="red"),
        _row(7, phase="drift_piece_a_p4_confirmed", decision="green"),
    ]
    _patch_runtime(monkeypatch, rows)

    summary = observer.run_observation(_args(tmp_path, thresholds=_thresholds(tmp_path)))

    windows = [
        json.loads(line)
        for line in Path(str(summary["windows_path"])).read_text(encoding="utf-8").splitlines()
    ]
    assert [window["window_complete"] for window in windows] == [True, True, True, False]
    assert windows[-1]["metrics"]["window_events"] == 1
    assert summary["drift_status"] == "confirmed"
    assert summary["trigger_lifecycle"] is True
    assert summary["first_confirmed_window_index"] == 3
    assert summary["last_complete_window_status"] == "confirmed"
