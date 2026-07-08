from __future__ import annotations

import csv
from pathlib import Path

from scripts.build_feature_ae_datasets import main as build_feature_ae_datasets


OUTPUT_DIR = Path("data/model_datasets")
MVP = OUTPUT_DIR / "feature_ae_good_mvp_v001.csv"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def test_materialized_feature_ae_manifests_exist_and_are_good_only() -> None:
    assert MVP.exists()

    rows = _rows(MVP)
    assert rows
    assert {row["dataset_version"] for row in rows} == {"feature_ae_good_mvp_v001"}
    assert {row["oracle_verdict"] for row in rows} == {"conforme"}
    assert {row["train_eligible"].lower() for row in rows} == {"true"}
    assert {row["train_eligibility_source"] for row in rows} == {"oracle_gt"}
    assert {row["is_defective"].lower() for row in rows} == {"false"}
    assert {row["quarantine_reason"] for row in rows} == {""}
    assert all("validation_set_replay_representative_v001" not in row["split_set"] for row in rows)
    assert all("calibration_good_reference_v001" not in row["split_set"] for row in rows)


def test_materialized_feature_ae_manifests_are_reproducible() -> None:
    before = MVP.read_text(encoding="utf-8")

    build_feature_ae_datasets([])

    after = MVP.read_text(encoding="utf-8")
    assert after == before
