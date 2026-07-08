from __future__ import annotations

import csv
from types import SimpleNamespace

import numpy as np
from PIL import Image

from scripts.run_drift_observation_replay import (
    RoiMaskReference,
    _select_observation_rows,
    _write_balanced_context_manifests,
)
from scripts.run_monitoring import evaluate_drift_metrics


def _mask(path, top: int, left: int, size: int = 8) -> str:
    array = np.zeros((32, 32), dtype=np.uint8)
    array[top : top + size, left : left + size] = 255
    Image.fromarray(array).save(path)
    return str(path)


def _event(mask_path: str, *, phase: str = "stable_piece_b", roi_ratio: float = 0.25):
    return SimpleNamespace(
        roi_mask_path=mask_path,
        relative_path="piece_b/test/good/example_1_1.jpg",
        source_class="piece_b",
        scenario_phase=phase,
        roi_ratio=roi_ratio,
    )


def test_roi_reference_keeps_piece_b_stable_under_novelty_threshold(tmp_path) -> None:
    reference = RoiMaskReference()
    reference.add_events(
        [
            _event(_mask(tmp_path / "b1.png", 8, 8)),
            _event(_mask(tmp_path / "b2.png", 9, 8)),
            _event(_mask(tmp_path / "b3.png", 8, 9)),
        ]
    )

    metrics = reference.metrics(
        [_event(_mask(tmp_path / "b4.png", 8, 8)), _event(_mask(tmp_path / "b5.png", 9, 8))],
        novelty_distance=0.50,
    )

    assert metrics["roi_mask_novelty_rate"] == 0.0
    assert metrics["roi_mask_nn_distance"] < 0.50


def test_roi_reference_flags_p4_masks_as_novel(tmp_path) -> None:
    reference = RoiMaskReference()
    reference.add_events(
        [
            _event(_mask(tmp_path / "b1.png", 4, 4)),
            _event(_mask(tmp_path / "b2.png", 5, 4)),
            _event(_mask(tmp_path / "b3.png", 4, 5)),
        ]
    )

    p4_events = [
        _event(_mask(tmp_path / "p4_1.png", 20, 20), phase="drift_piece_a_p4_confirmed", roi_ratio=0.51),
        _event(_mask(tmp_path / "p4_2.png", 21, 20), phase="drift_piece_a_p4_confirmed", roi_ratio=0.50),
    ]
    metrics = reference.metrics(p4_events, novelty_distance=0.50)

    assert metrics["roi_mask_novelty_rate"] == 1.0
    assert metrics["roi_mask_nn_distance"] >= 0.50
    assert metrics["roi_area_ratio"] >= 0.50


def test_one_complete_p4_window_confirms_roi_drift() -> None:
    result = evaluate_drift_metrics(
        scenario_id="production_replay_natural_piece_b_to_piece_a_p4_drift",
        window_events=10,
        domain_ratio=0.0,
        roi_mask_nn_distance=0.74,
        roi_mask_novelty_rate=1.0,
        roi_area_ratio=0.51,
        context_events_total=10,
        alert_rate=0.0,
        red_rate=0.0,
        roi_fail_rate=0.0,
        oracle_fn_rate=0.0,
        critical_window_count=0,
        thresholds={
            "drift": {
                "min_window_events": 10,
                "confirm_windows": 2,
                "roi_mask_novelty_rate_critical": 0.8,
                "roi_mask_nn_distance_critical": 0.5,
                "roi_area_ratio_critical": 0.4,
            }
        },
    )

    assert result["status"] == "confirmed"
    assert result["drift_confirmed"] is True
    assert result["critical_window_count"] == 1


def test_progressive_p4_window_suspects_before_confirming() -> None:
    mixed = evaluate_drift_metrics(
        scenario_id="production_replay_natural_piece_b_to_piece_a_p4_drift",
        window_events=10,
        domain_ratio=0.40,
        roi_mask_nn_distance=0.74,
        roi_mask_novelty_rate=0.40,
        roi_area_ratio=0.51,
        context_events_total=4,
        alert_rate=0.0,
        red_rate=0.0,
        roi_fail_rate=0.0,
        oracle_fn_rate=0.0,
        critical_window_count=0,
        thresholds={
            "drift": {
                "min_window_events": 10,
                "confirm_windows": 2,
                "roi_mask_novelty_rate_critical": 0.8,
                "roi_mask_nn_distance_critical": 0.5,
                "roi_area_ratio_critical": 0.4,
                "domain_ratio_critical": 0.4,
            }
        },
    )

    confirmed = evaluate_drift_metrics(
        scenario_id="production_replay_natural_piece_b_to_piece_a_p4_drift",
        window_events=10,
        domain_ratio=1.0,
        roi_mask_nn_distance=0.74,
        roi_mask_novelty_rate=1.0,
        roi_area_ratio=0.51,
        context_events_total=10,
        alert_rate=0.0,
        red_rate=0.0,
        roi_fail_rate=0.0,
        oracle_fn_rate=0.0,
        critical_window_count=int(mixed["critical_window_count"]),
        thresholds={
            "drift": {
                "min_window_events": 10,
                "confirm_windows": 2,
                "roi_mask_novelty_rate_critical": 0.8,
                "roi_mask_nn_distance_critical": 0.5,
                "roi_area_ratio_critical": 0.4,
                "domain_ratio_critical": 0.4,
            }
        },
    )

    assert mixed["status"] == "suspected"
    assert mixed["drift_confirmed"] is False
    assert mixed["critical_window"] is False
    assert confirmed["status"] == "confirmed"
    assert confirmed["drift_confirmed"] is True


def test_drift_context_writes_balanced_train_anchor_and_eval_manifests(tmp_path) -> None:
    context_rows = [
        {"scenario_phase": "drift_piece_a_p4_suspected", "relative_path": f"p4_good_{index}.jpg", "label": "good", "is_defective": "False"}
        for index in range(3)
    ] + [
        {"scenario_phase": "drift_piece_a_p4_confirmed", "relative_path": "p4_defective.jpg", "label": "defective", "is_defective": "True"}
    ]
    stable_rows = [
        {
            "scenario_phase": "stable_baseline_piece_b",
            "relative_path": f"{view_pair}_{index}.jpg",
            "view_pairs": view_pair,
            "label": "good",
            "is_defective": "False",
        }
        for view_pair in ["Casting_class1:1_2", "Casting_class1:1_3", "Casting_class1:2_3"]
        for index in range(10)
    ]

    anchor_path = tmp_path / "anchor.csv"
    eval_path = tmp_path / "eval.csv"
    stats = _write_balanced_context_manifests(
        anchor_path=anchor_path,
        eval_path=eval_path,
        context_rows=context_rows,
        stable_rows=stable_rows,
    )

    with anchor_path.open(newline="", encoding="utf-8") as file:
        anchor_rows = list(csv.DictReader(file))
    with eval_path.open(newline="", encoding="utf-8") as file:
        eval_rows = list(csv.DictReader(file))

    assert stats["train_p4_good_count"] == 3
    assert stats["train_piece_b_anchor_count"] == 9
    assert stats["train_p1_anchor_count"] == 3
    assert stats["train_p2_anchor_count"] == 3
    assert stats["train_p3_anchor_count"] == 3
    assert stats["eval_p4_count"] == 4
    assert stats["eval_piece_b_count"] == 12
    assert stats["eval_p1_count"] == 4
    assert stats["eval_p2_count"] == 4
    assert stats["eval_p3_count"] == 4
    assert len(anchor_rows) == 9
    assert len(eval_rows) == 16


def test_short_drift_observation_selects_reference_then_mixed_then_full_p4_window() -> None:
    rows = [
        {"scenario_phase": "stable_baseline_piece_b", "sequence_number": str(index)}
        for index in range(330)
    ] + [
        {"scenario_phase": "drift_piece_a_p4_suspected", "sequence_number": str(index)}
        for index in range(30)
    ] + [
        {"scenario_phase": "drift_piece_a_p4_confirmed", "sequence_number": str(index)}
        for index in range(33)
    ]

    selected = _select_observation_rows(
        rows,
        stable_reference_events=10,
        drift_observation_windows=3,
        window_size=10,
    )

    assert len(selected) == 40
    assert [row["scenario_phase"] for row in selected[:10]] == ["stable_baseline_piece_b"] * 10
    assert sum(1 for row in selected[10:20] if row["scenario_phase"] == "stable_baseline_piece_b") == 8
    assert sum(1 for row in selected[10:20] if row["scenario_phase"] != "stable_baseline_piece_b") == 2
    assert [row["scenario_phase"] for row in selected[20:30]] == ["drift_piece_a_p4_suspected"] * 10
    assert [row["scenario_phase"] for row in selected[30:]] == ["drift_piece_a_p4_suspected"] * 10
