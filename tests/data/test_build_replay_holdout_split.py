from __future__ import annotations

import csv
from pathlib import Path

from scripts.build_replay_holdout_split import build_split


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _ids(rows: list[dict[str, str]], key: str) -> set[str]:
    return {row[key] for row in rows}


def test_build_replay_holdout_split_is_deterministic(tmp_path: Path) -> None:
    train_output = tmp_path / "train.csv"
    gate_output = tmp_path / "gate.csv"

    train_rows, gate_rows = build_split(train_output=train_output, gate_output=gate_output)

    assert _read_rows(train_output) == train_rows
    assert _read_rows(gate_output) == gate_rows
    assert len(train_rows) == 432
    assert len(gate_rows) == 130
    assert sum(row["is_defective"].lower() == "true" for row in train_rows) == 16
    assert sum(row["is_defective"].lower() == "true" for row in gate_rows) == 10
    assert _ids(train_rows, "source_event_id").isdisjoint(_ids(gate_rows, "event_id"))
    assert {row["scenario_id"] for row in train_rows} == {"production_replay_natural_train_v004"}
    assert {row["validation_set_id"] for row in gate_rows} == {"validation_set_replay_gate_v003"}
