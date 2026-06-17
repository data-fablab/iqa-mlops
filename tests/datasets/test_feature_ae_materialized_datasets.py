from __future__ import annotations

import csv
from pathlib import Path

from scripts.build_feature_ae_datasets import main as build_feature_ae_datasets


OUTPUT_DIR = Path("data/model_datasets")
V002 = OUTPUT_DIR / "feature_ae_good_v002.csv"
V003 = OUTPUT_DIR / "feature_ae_good_v003.csv"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def test_materialized_feature_ae_manifests_exist_and_are_good_only() -> None:
    assert V002.exists()
    assert V003.exists()

    for path, dataset_version, manifest_version in [
        (V002, "feature_ae_good_v002", "feature_ae_good_v002_manifest_v001"),
        (V003, "feature_ae_good_v003", "feature_ae_good_v003_manifest_v001"),
    ]:
        rows = _rows(path)
        assert rows
        assert {row["dataset_version"] for row in rows} == {dataset_version}
        assert {row["manifest_version"] for row in rows} == {manifest_version}
        assert {row["oracle_verdict"] for row in rows} == {"conforme"}
        assert {row["train_eligible"] for row in rows} == {"true"}
        assert {row["train_eligibility_source"] for row in rows} == {"oracle_gt"}
        assert {row["is_defective"] for row in rows} == {"False"}
        assert {row["quarantine_reason"] for row in rows} == {""}
        assert all("validation_set_v001" not in row["split_set"] for row in rows)
        assert all("calibration_set_v001" not in row["split_set"] for row in rows)


def test_materialized_feature_ae_manifests_are_reproducible() -> None:
    before = {path: path.read_text(encoding="utf-8") for path in [V002, V003]}

    build_feature_ae_datasets([])

    after = {path: path.read_text(encoding="utf-8") for path in [V002, V003]}
    assert after == before
