from __future__ import annotations

import csv
from pathlib import Path

from scripts.build_validation_gate_v2 import build_gate_v2


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_build_validation_gate_v2_is_deterministic(tmp_path: Path) -> None:
    output = tmp_path / "validation_set_replay_gate_v002.csv"

    rows = build_gate_v2(output_path=output)
    persisted = _read_rows(output)

    assert persisted == rows
    assert len(rows) == 134
    assert sum(row["is_defective"].lower() == "false" for row in rows) == 120
    assert sum(row["is_defective"].lower() == "true" for row in rows) == 14
    assert {row["validation_set_id"] for row in rows} == {"validation_set_replay_gate_v002"}
    assert any(int(row["n_images"]) > 1 for row in rows)
