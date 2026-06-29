"""Build a tiny piece-B replay scenario for local MLOps smoke tests."""

from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

from iqa.metadata.contracts import (
    BOOTSTRAP_COLUMNS,
    RAW_DATASET_ID,
    REPLAY_COLUMNS,
    VALIDATION_COLUMNS,
)


SOURCE_MANIFEST = Path("data/metadata/casting_piece_events.csv")
BOOTSTRAP_MANIFEST = Path("data/metadata/feature_ae_bootstrap_piece_b_minimal_v001.csv")
VALIDATION_MANIFEST = Path("data/validation/validation_set_piece_b_minimal_v001.csv")
REPLAY_MANIFEST = Path("data/metadata/casting_flux_replay_plan_piece_b_minimal_v001.csv")
SCENARIOS_MANIFEST = Path("data/metadata/replay_scenarios.csv")

PIECE_B_VIEW_PAIRS = "Casting_class1:1_2|Casting_class1:1_3|Casting_class1:2_3"
SCENARIO_ID = "production_replay_natural_piece_b_minimal"
BOOTSTRAP_DATASET_VERSION = "feature_ae_piece_b_minimal_bootstrap_v001"
VALIDATION_SET_ID = "validation_set_piece_b_minimal_v001"
SCENARIO_VERSION = "casting_flux_replay_plan_piece_b_minimal_v001"
ROI_MODEL_VERSION = "roi_segmenter_v001_fixed"
FEATURE_AE_VERSION = "rd_feature_ae_gated_v001_bootstrap"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_rows(path: Path, fieldnames: tuple[str, ...] | list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def piece_b_rows() -> list[dict[str, str]]:
    rows = [
        row
        for row in read_rows(SOURCE_MANIFEST)
        if row["source_class"] == "Casting_class1"
        and row["view_pairs"] == PIECE_B_VIEW_PAIRS
        and row["n_images"] == "3"
    ]
    rows.sort(key=lambda row: (row["source_timestamp"], row["event_id"]))
    return rows


def select_rows(rows: list[dict[str, str]], count: int, used_ids: set[str], **filters: str) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if row["event_id"] not in used_ids
        and all(str(row[key]).lower() == str(value).lower() for key, value in filters.items())
    ][:count]
    if len(selected) != count:
        raise RuntimeError(f"expected {count} rows for {filters}, got {len(selected)}")
    used_ids.update(row["event_id"] for row in selected)
    return selected


def with_metadata(row: dict[str, str], *, manifest_id: str, dataset_version: str) -> dict[str, str]:
    enriched = dict(row)
    enriched.update(
        {
            "raw_dataset_id": RAW_DATASET_ID,
            "manifest_id": manifest_id,
            "piece_event_id": row["event_id"],
            "dataset_version": dataset_version,
            "replay_id": "",
            "validation_id": "",
            "scenario_version": "",
        }
    )
    return enriched


def build_bootstrap(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output = []
    for index, row in enumerate(rows, start=1):
        enriched = with_metadata(
            row,
            manifest_id="feature_ae_bootstrap_piece_b_minimal_v001",
            dataset_version=BOOTSTRAP_DATASET_VERSION,
        )
        enriched.update(
            {
                "bootstrap_dataset_version": BOOTSTRAP_DATASET_VERSION,
                "bootstrap_sequence": str(index),
                "bootstrap_role": "train_normal_piece_b_minimal",
            }
        )
        output.append(enriched)
    return output


def build_validation(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output = []
    for row in rows:
        enriched = with_metadata(
            row,
            manifest_id=VALIDATION_SET_ID,
            dataset_version=VALIDATION_SET_ID,
        )
        enriched.update(
            {
                "validation_set_id": VALIDATION_SET_ID,
                "validation_role": "mvp_piece_b_minimal_gate",
                "validation_id": VALIDATION_SET_ID,
                "scenario_version": "feature_ae_piece_b_minimal_v001",
            }
        )
        output.append(enriched)
    return output


def simulated_event_id(source_event_id: str) -> str:
    digest = hashlib.sha1(f"{SCENARIO_ID}:{source_event_id}".encode("utf-8")).hexdigest()[:12]
    return f"sim_event_{digest}"


def build_replay(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output = []
    base_time = datetime(2026, 6, 26, 9, 0, 0)
    for index, row in enumerate(rows, start=1):
        event_time = base_time + timedelta(minutes=5 * (index - 1))
        event_time_text = event_time.isoformat(timespec="seconds")
        simulated_id = simulated_event_id(row["event_id"])
        output.append(
            {
                "simulated_event_id": simulated_id,
                "scenario_id": SCENARIO_ID,
                "scenario_type": "production",
                "scenario_phase": "natural_replay_piece_b_minimal_v001",
                "is_representative": "False",
                "sequence_number": str(index),
                "scheduled_at": event_time_text,
                "production_date": event_time.date().isoformat(),
                "shift_id": "day_shift",
                "station_id": "QC-CASTING-01",
                "lot_id": "IQA-PIECEB-MIN-L001",
                "sequence_in_lot": str(index),
                "piece_serial": f"IQA-PIECEB-MIN-L001-P{index:03d}",
                "source_event_id": row["event_id"],
                "source_event_key": row["event_key"],
                "source_class": row["source_class"],
                "source_group_key": row["group_key"],
                "source_timestamp": row["source_timestamp"],
                "label": row["label"],
                "is_defective": row["is_defective"],
                "expected_review_required": row["is_defective"],
                "n_images": row["n_images"],
                "source_classes": row["source_classes"],
                "view_pairs": row["view_pairs"],
                "image_ids": row["image_ids"],
                "relative_paths": row["relative_paths"],
                "event_time": event_time_text,
                "recorded_at": "2026-06-26T00:00:00",
                "is_simulated": "True",
                "roi_model_version": ROI_MODEL_VERSION,
                "feature_ae_version": FEATURE_AE_VERSION,
                "raw_dataset_id": RAW_DATASET_ID,
                "manifest_id": SCENARIO_VERSION,
                "piece_event_id": simulated_id,
                "dataset_version": SCENARIO_ID,
                "replay_id": SCENARIO_ID,
                "validation_id": "",
                "scenario_version": SCENARIO_VERSION,
            }
        )
    return output


def upsert_scenario() -> None:
    rows = read_rows(SCENARIOS_MANIFEST)
    rows = [row for row in rows if row["scenario_id"] != SCENARIO_ID]
    rows.append(
        {
            "scenario_id": SCENARIO_ID,
            "scenario_type": "production",
            "purpose": "Replay natural minimal piece B class1 pour smoke MLOps local.",
            "is_representative": "false",
            "output_path": str(REPLAY_MANIFEST).replace("\\", "/"),
            "ordering_rule": "Casting_class1 piece B complete uniquement; bootstrap 2 events; validation/replay disjoints",
            "model_lifecycle_use": "smoke MLOps local bootstrap validation replay",
        }
    )
    write_rows(SCENARIOS_MANIFEST, list(rows[0].keys()), rows)


def main() -> None:
    rows = piece_b_rows()
    used_ids: set[str] = set()

    bootstrap = select_rows(rows, 2, used_ids, split_set="train", label="good", is_defective="False")
    validation_good = select_rows(rows, 2, used_ids, split_set="test", label="good", is_defective="False")
    validation_defective = select_rows(rows, 2, used_ids, split_set="test", label="defective", is_defective="True")
    replay_good = select_rows(rows, 6, used_ids, label="good", is_defective="False")
    replay_defective = select_rows(rows, 2, used_ids, label="defective", is_defective="True")

    write_rows(BOOTSTRAP_MANIFEST, (*BOOTSTRAP_COLUMNS, "raw_dataset_id", "manifest_id", "piece_event_id", "dataset_version", "replay_id", "validation_id", "scenario_version"), build_bootstrap(bootstrap))
    write_rows(VALIDATION_MANIFEST, (*VALIDATION_COLUMNS, "raw_dataset_id", "manifest_id", "piece_event_id", "dataset_version", "replay_id", "validation_id", "scenario_version"), build_validation([*validation_good, *validation_defective]))
    write_rows(REPLAY_MANIFEST, (*REPLAY_COLUMNS, "raw_dataset_id", "manifest_id", "piece_event_id", "dataset_version", "replay_id", "validation_id", "scenario_version"), build_replay([*replay_good, *replay_defective]))
    upsert_scenario()

    print(
        {
            "bootstrap": len(bootstrap),
            "validation": len(validation_good) + len(validation_defective),
            "replay": len(replay_good) + len(replay_defective),
            "scenario_id": SCENARIO_ID,
        }
    )


if __name__ == "__main__":
    main()
