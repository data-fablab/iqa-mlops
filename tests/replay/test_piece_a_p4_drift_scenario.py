from __future__ import annotations

import csv
from pathlib import Path


SCENARIO_ID = "production_replay_natural_piece_b_to_piece_a_p4_drift"
PLAN = Path("data/metadata/casting_flux_replay_plan_piece_b_to_piece_a_p4_drift_v001.csv")
CORRECTION_VALIDATION = Path("data/validation/validation_set_piece_b_to_piece_a_p4_drift_v001.csv")
CORRECTION_GT_MASKS = Path("data/validation/validation_gt_masks_piece_b_to_piece_a_p4_drift_v001.csv")
RAW_SOURCE_ROOT = Path("data/raw/hss-iad")
PIECE_A_P4_VIEW_PAIRS = "Casting_class1:2_3"


def _rows() -> list[dict[str, str]]:
    with PLAN.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _chunks(rows: list[dict[str, str]], size: int) -> list[list[dict[str, str]]]:
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def test_piece_a_p4_drift_plan_contains_piece_b_baseline_then_p4() -> None:
    rows = _rows()

    assert rows
    assert {row["scenario_id"] for row in rows} == {SCENARIO_ID}
    assert rows[0]["scenario_phase"] == "stable_baseline_piece_b"
    assert rows[0]["source_class"] == "Casting_class1"
    stable_rows = [row for row in rows if row["scenario_phase"] == "stable_baseline_piece_b"]
    assert stable_rows
    assert all(row["is_defective"].lower() != "true" for row in stable_rows)
    assert len(stable_rows) % 30 == 0
    p4_rows = [
        row
        for row in rows
        if row["source_class"] == "Casting_class1" and row["scenario_phase"] != "stable_baseline_piece_b"
    ]
    assert p4_rows
    assert {row["scenario_phase"] for row in p4_rows} == {
        "drift_piece_a_p4_suspected",
        "drift_piece_a_p4_confirmed",
        "correction_replay",
    }
    assert {row["view_pairs"] for row in p4_rows} == {PIECE_A_P4_VIEW_PAIRS}
    assert len(rows) == 423


def test_piece_a_p4_rows_are_not_forced_defective() -> None:
    rows = _rows()
    p4_rows = [
        row
        for row in rows
        if row["source_class"] == "Casting_class1" and row["scenario_phase"] != "stable_baseline_piece_b"
    ]

    assert {row["label"] for row in p4_rows} == {"good", "defective"}
    assert sum(row["label"] == "defective" for row in p4_rows) == 12
    assert sum(row["label"] == "good" for row in p4_rows) == 81


def test_piece_a_p4_drift_plan_has_two_complete_critical_windows() -> None:
    rows = _rows()
    critical_window_indexes = []

    for window_index, window_rows in enumerate(_chunks(rows, 30), start=1):
        if len(window_rows) < 30:
            continue
        p4_count = sum(row["scenario_phase"] != "stable_baseline_piece_b" for row in window_rows)
        stable_defective_count = sum(
            row["scenario_phase"] == "stable_baseline_piece_b" and row["is_defective"].lower() == "true"
            for row in window_rows
        )
        domain_ratio = p4_count / len(window_rows)
        assert stable_defective_count == 0
        if domain_ratio >= 0.50:
            critical_window_indexes.append(window_index)

    assert any(
        current + 1 == following
        for current, following in zip(critical_window_indexes, critical_window_indexes[1:], strict=False)
    )


def test_classification_selection_manifest_uses_p4_observation_windows() -> None:
    path = Path("data/validation/classification_selection_piece_b_to_piece_a_p4_drift_v001.csv")
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    assert rows
    assert {row["validation_role"] for row in rows} == {"classification_selection_piece_a_p4"}
    assert {row["scenario_phase"] for row in rows} == {
        "drift_piece_a_p4_suspected",
        "drift_piece_a_p4_confirmed",
    }
    assert {row["view_pairs"] for row in rows} == {PIECE_A_P4_VIEW_PAIRS}


def test_piece_a_p4_rows_exclude_incomplete_piece_b_events() -> None:
    rows = _rows()
    p4_rows = [
        row
        for row in rows
        if row["source_class"] == "Casting_class1" and row["scenario_phase"] != "stable_baseline_piece_b"
    ]

    assert not {
        "2022-02-18_13_53_57_323",
        "2022-02-18_13_59_42_257",
        "2022-02-18_16_28_25_855",
        "2022-02-18_16_44_47_129",
        "2022-02-18_16_57_39_346",
        "2022-02-19_09_07_21_691",
        "2022-02-19_09_17_25_745",
        "2022-02-19_09_37_52_865",
    }.intersection({row["source_group_key"] for row in p4_rows})


def test_piece_a_p4_correction_gate_manifest_contains_piece_b_and_p4() -> None:
    with CORRECTION_VALIDATION.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    roles = {row["validation_role"] for row in rows}
    assert roles == {"piece_b_non_regression_gate", "piece_a_p4_drift_correction_gate"}
    p4_rows = [row for row in rows if row["validation_role"] == "piece_a_p4_drift_correction_gate"]
    piece_b_rows = [row for row in rows if row["validation_role"] == "piece_b_non_regression_gate"]
    assert len(piece_b_rows) >= 8
    assert len(p4_rows) == 85
    assert {row["view_pairs"] for row in p4_rows} == {PIECE_A_P4_VIEW_PAIRS}
    assert {row["label"] for row in p4_rows} == {"good", "defective"}


def test_piece_a_p4_gt_masks_exist_in_canonical_raw_source() -> None:
    with CORRECTION_GT_MASKS.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    assert rows
    missing = [row["gt_mask_path"] for row in rows if not (RAW_SOURCE_ROOT / row["gt_mask_path"]).is_file()]
    assert missing == []
