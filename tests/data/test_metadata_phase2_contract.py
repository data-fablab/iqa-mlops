from __future__ import annotations

import csv
from pathlib import Path

from iqa.metadata.contracts import MANIFEST_CONTRACTS, PHASE2_METADATA_COLUMNS


ROOT = Path(".")
DATA_CONTRACTS_DOC = ROOT / "docs" / "data-contracts.md"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def test_phase2_metadata_columns_and_historical_columns_exist() -> None:
    for contract in MANIFEST_CONTRACTS.values():
        rows = _read_csv(ROOT / contract.path)
        assert rows, contract.key
        columns = rows[0].keys()

        assert all(column in columns for column in contract.historical_columns), contract.key
        assert all(column in columns for column in PHASE2_METADATA_COLUMNS), contract.key


def test_phase2_metadata_values_are_stable_by_manifest() -> None:
    for contract in MANIFEST_CONTRACTS.values():
        rows = _read_csv(ROOT / contract.path)

        assert {row["raw_dataset_id"] for row in rows} == {"hss_iad_casting_raw_v1"}
        assert {row["manifest_id"] for row in rows} == {contract.manifest_id}
        assert {row["dataset_version"] for row in rows} == {contract.dataset_version}
        assert {row["replay_id"] for row in rows} == {contract.replay_id}
        assert {row["validation_id"] for row in rows} == {contract.validation_id}
        assert {row["scenario_version"] for row in rows} == {contract.scenario_version}


def test_piece_event_id_identity_rules_are_enforced() -> None:
    for contract in MANIFEST_CONTRACTS.values():
        rows = _read_csv(ROOT / contract.path)
        piece_event_ids = [row["piece_event_id"] for row in rows]

        assert all(piece_event_ids), contract.key
        assert len(piece_event_ids) == len(set(piece_event_ids)), contract.key
        assert all(row["piece_event_id"] == row[contract.identity_column] for row in rows), contract.key


def test_replay_metadata_keeps_source_identity_and_existing_runtime_values() -> None:
    for key in ["natural_replay", "drift_replay"]:
        contract = MANIFEST_CONTRACTS[key]
        rows = _read_csv(ROOT / contract.path)

        assert contract.scenario_id is not None
        assert {row["scenario_id"] for row in rows} == {contract.scenario_id}
        assert {row["dataset_version"] for row in rows} == {contract.dataset_version}
        assert all(row["source_event_id"] for row in rows)
        assert all(row["event_time"] for row in rows)
        assert all(row["recorded_at"] for row in rows)
        assert {row["is_simulated"].lower() for row in rows} == {"true"}
        assert all(row["piece_event_id"] == row["simulated_event_id"] for row in rows)


def test_data_contracts_document_canonical_identifiers_and_manifests() -> None:
    content = DATA_CONTRACTS_DOC.read_text(encoding="utf-8")

    for column in PHASE2_METADATA_COLUMNS:
        assert column in content
    for identifier in ["source_event_id", "scenario_id", "lot_id", "prediction_id", "model_version", "feedback_id"]:
        assert identifier in content
    for contract in MANIFEST_CONTRACTS.values():
        assert contract.manifest_id in content
        assert str(contract.path).replace("\\", "/") in content
    assert "sha256 -> piece_event -> scenario -> lot -> dataset_version -> model_version -> prediction -> feedback" in content
