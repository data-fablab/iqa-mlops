"""Build a full piece-B natural replay scenario for local lifecycle promotion tests."""

from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

from iqa.metadata.contracts import RAW_DATASET_ID, REPLAY_COLUMNS, VALIDATION_COLUMNS


SOURCE_MANIFEST = Path("data/metadata/casting_piece_events.csv")
VALIDATION_MANIFEST = Path("data/validation/validation_set_piece_b_full_v001.csv")
VALIDATION_GT_MASKS_MANIFEST = Path("data/validation/validation_gt_masks_piece_b_full_v001.csv")
REPLAY_MANIFEST = Path("data/metadata/casting_flux_replay_plan_piece_b_full_v001.csv")
SCENARIOS_MANIFEST = Path("data/metadata/replay_scenarios.csv")

PIECE_B_VIEW_PAIRS = "Casting_class1:1_2|Casting_class1:1_3|Casting_class1:2_3"
SCENARIO_ID = "production_replay_natural_piece_b_full"
VALIDATION_SET_ID = "validation_set_piece_b_full_v001"
SCENARIO_VERSION = "casting_flux_replay_plan_piece_b_full_v001"
ROI_MODEL_VERSION = "roi_segmenter_v001_fixed"
FEATURE_AE_VERSION = "rd_feature_ae_gated_v001_bootstrap"
PIECES_PER_LOT = 12


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


def split_pipe(value: str) -> list[str]:
    return [item.strip() for item in value.split("|") if item.strip()]


def mask_paths_for(row: dict[str, str]) -> str:
    if row["is_defective"].lower() != "true":
        return ""
    masks = []
    for relative_path in split_pipe(row["relative_paths"]):
        path = Path(relative_path.replace("\\", "/"))
        mask_name = f"{path.stem}_mask.png"
        masks.append(f"Casting_class1/ground_truth/defective/{mask_name}")
    return "|".join(masks)


def with_metadata(row: dict[str, str], *, manifest_id: str, dataset_version: str) -> dict[str, str]:
    enriched = dict(row)
    enriched.update(
        {
            "raw_dataset_id": RAW_DATASET_ID,
            "manifest_id": manifest_id,
            "piece_event_id": row["event_id"],
            "dataset_version": dataset_version,
            "replay_id": SCENARIO_ID,
            "validation_id": "",
            "scenario_version": SCENARIO_VERSION,
            "gt_mask_paths": mask_paths_for(row),
        }
    )
    return enriched


def build_validation(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output = []
    for row in rows:
        enriched = with_metadata(row, manifest_id=VALIDATION_SET_ID, dataset_version=VALIDATION_SET_ID)
        enriched.update(
            {
                "validation_set_id": VALIDATION_SET_ID,
                "validation_role": "mvp_piece_b_full_gate",
                "validation_id": VALIDATION_SET_ID,
            }
        )
        output.append(enriched)
    return output


def simulated_event_id(source_event_id: str, view_index: int) -> str:
    digest = hashlib.sha1(f"{SCENARIO_ID}:{source_event_id}:{view_index}".encode("utf-8")).hexdigest()[:12]
    return f"sim_event_{digest}"


def build_replay(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output = []
    base_time = datetime(2026, 6, 26, 10, 0, 0)
    sequence_number = 0
    for piece_index, row in enumerate(rows, start=1):
        lot_index = (piece_index - 1) // PIECES_PER_LOT + 1
        lot_id = f"IQA-PIECEB-FULL-L{lot_index:03d}"
        image_ids = split_pipe(row["image_ids"])
        relative_paths = split_pipe(row["relative_paths"])
        view_pairs = split_pipe(row["view_pairs"])
        gt_mask_paths = split_pipe(mask_paths_for(row))
        for view_index, relative_path in enumerate(relative_paths, start=1):
            sequence_number += 1
            event_time = base_time + timedelta(minutes=2 * (sequence_number - 1))
            event_time_text = event_time.isoformat(timespec="seconds")
            simulated_id = simulated_event_id(row["event_id"], view_index)
            output.append(
                {
                    "simulated_event_id": simulated_id,
                    "scenario_id": SCENARIO_ID,
                    "scenario_type": "production",
                    "scenario_phase": "natural_replay_piece_b_full_v001",
                    "is_representative": "False",
                    "sequence_number": str(sequence_number),
                    "scheduled_at": event_time_text,
                    "production_date": event_time.date().isoformat(),
                    "shift_id": "day_shift",
                    "station_id": "QC-CASTING-01",
                    "lot_id": lot_id,
                    "sequence_in_lot": str((piece_index - 1) % PIECES_PER_LOT + 1),
                    "piece_serial": f"{lot_id}-P{(piece_index - 1) % PIECES_PER_LOT + 1:03d}-V{view_index}",
                    "source_event_id": row["event_id"],
                    "source_event_key": f"{row['event_key']}#v{view_index}",
                    "source_class": row["source_class"],
                    "source_group_key": row["group_key"],
                    "source_timestamp": row["source_timestamp"],
                    "label": row["label"],
                    "is_defective": row["is_defective"],
                    "expected_review_required": row["is_defective"],
                    "n_images": "1",
                    "source_classes": row["source_classes"],
                    "view_pairs": view_pairs[view_index - 1],
                    "image_ids": image_ids[view_index - 1],
                    "relative_paths": relative_path,
                    "event_time": event_time_text,
                    "recorded_at": "2026-06-26T00:00:00",
                    "is_simulated": "True",
                    "roi_model_version": ROI_MODEL_VERSION,
                    "feature_ae_version": FEATURE_AE_VERSION,
                    "raw_dataset_id": RAW_DATASET_ID,
                    "manifest_id": SCENARIO_VERSION,
                    "piece_event_id": row["event_id"],
                    "dataset_version": SCENARIO_ID,
                    "replay_id": SCENARIO_ID,
                    "validation_id": "",
                    "scenario_version": SCENARIO_VERSION,
                    "gt_mask_paths": gt_mask_paths[view_index - 1] if gt_mask_paths else "",
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
            "purpose": "Replay natural complet Piece B class1 pour validation des promotions candidates.",
            "is_representative": "false",
            "output_path": str(REPLAY_MANIFEST).replace("\\", "/"),
            "ordering_rule": "Casting_class1 piece B complete; 3 vues rejouees; lots de 12 pieces.",
            "model_lifecycle_use": "promotion progressive Feature-AE sur replay natural local",
        }
    )
    write_rows(SCENARIOS_MANIFEST, list(rows[0].keys()), rows)


def main() -> None:
    rows = piece_b_rows()
    validation_rows = [row for row in rows if row["split_set"] == "test"]
    replay_rows = build_replay(rows)

    write_rows(
        VALIDATION_MANIFEST,
        (*VALIDATION_COLUMNS, "raw_dataset_id", "manifest_id", "piece_event_id", "dataset_version", "replay_id", "validation_id", "scenario_version", "gt_mask_paths"),
        build_validation(validation_rows),
    )
    write_rows(VALIDATION_GT_MASKS_MANIFEST, ["image_id", "gt_mask_path"], [])
    write_rows(
        REPLAY_MANIFEST,
        (*REPLAY_COLUMNS, "raw_dataset_id", "manifest_id", "piece_event_id", "dataset_version", "replay_id", "validation_id", "scenario_version", "gt_mask_paths"),
        replay_rows,
    )
    upsert_scenario()

    print(
        {
            "piece_events": len(rows),
            "replay_image_events": len(replay_rows),
            "test_validation_piece_events": len(validation_rows),
            "lots": len({row["lot_id"] for row in replay_rows}),
            "scenario_id": SCENARIO_ID,
        }
    )


if __name__ == "__main__":
    main()
