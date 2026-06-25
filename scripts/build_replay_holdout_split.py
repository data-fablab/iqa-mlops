"""Build scenario-B replay/train split and a fixed gate holdout panel."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from iqa.metadata.contracts import RAW_DATASET_ID, contract_for_key


SOURCE_REPLAY = Path("data/metadata/casting_flux_replay_plan_natural_v003.csv")
TRAIN_REPLAY_OUTPUT = Path("data/metadata/casting_flux_replay_plan_natural_train_v004.csv")
GATE_OUTPUT = Path("data/validation/validation_set_replay_gate_v003.csv")

TRAIN_SCENARIO_ID = "production_replay_natural_train_v004"
TRAIN_MANIFEST_ID = "casting_flux_replay_plan_natural_train_v004"
TRAIN_DATASET_VERSION = "production_replay_natural_train_v004"
GATE_ID = "validation_set_replay_gate_v003"
GATE_ROLE = "mvp_gate_replay_holdout_reference"
SCENARIO_VERSION = "feature_ae_mvp_v003"

HOLDOUT_DEFECT_TARGETS = {
    "Casting_class1": 3,
    "Casting_class2": 4,
    "Casting_class3": 3,
}
HOLDOUT_GOOD_TOTAL = 120


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fieldnames} for row in rows])


def is_defective(row: dict[str, str]) -> bool:
    return str(row.get("is_defective") or "").strip().lower() == "true"


def source_class(row: dict[str, str]) -> str:
    return str(row.get("source_class") or "")


def source_event_id(row: dict[str, str]) -> str:
    return str(row.get("source_event_id") or row.get("event_id") or "")


def spread_select(rows: list[dict[str, str]], target: int) -> list[dict[str, str]]:
    ordered = sorted(rows, key=lambda row: (row.get("source_timestamp") or "", source_event_id(row)))
    if target >= len(ordered):
        return ordered
    if target <= 0:
        return []
    if target == 1:
        return [ordered[len(ordered) // 2]]
    indexes = sorted(round(i * (len(ordered) - 1) / (target - 1)) for i in range(target))
    return [ordered[index] for index in indexes]


def good_targets(replay_rows: list[dict[str, str]], total: int) -> dict[str, int]:
    counts: Counter[str] = Counter(source_class(row) for row in replay_rows if not is_defective(row))
    observed_total = sum(counts.values())
    raw = {key: total * value / observed_total for key, value in counts.items()}
    targets = {key: int(value) for key, value in raw.items()}
    remainder = total - sum(targets.values())
    ranked = sorted(raw, key=lambda key: raw[key] - targets[key], reverse=True)
    for key in ranked[:remainder]:
        targets[key] += 1
    return targets


def select_holdout_rows(replay_rows: list[dict[str, str]], *, good_total: int) -> list[dict[str, str]]:
    by_class_label: dict[tuple[str, bool], list[dict[str, str]]] = defaultdict(list)
    for row in replay_rows:
        by_class_label[(source_class(row), is_defective(row))].append(row)

    holdout: list[dict[str, str]] = []
    for klass, target in sorted(HOLDOUT_DEFECT_TARGETS.items()):
        holdout.extend(spread_select(by_class_label[(klass, True)], target))
    for klass, target in sorted(good_targets(replay_rows, good_total).items()):
        holdout.extend(spread_select(by_class_label[(klass, False)], target))
    return sorted(holdout, key=lambda row: (is_defective(row), source_class(row), row.get("source_timestamp") or ""))


def to_gate_row(row: dict[str, str]) -> dict[str, str]:
    validation_fields = contract_for_key("validation_gate_set_v3").required_columns
    mapped = {field: "" for field in validation_fields}
    mapped.update(
        {
            "event_id": source_event_id(row),
            "event_key": row.get("source_event_key", ""),
            "source_class": row.get("source_class", ""),
            "group_key": row.get("source_group_key", ""),
            "source_timestamp": row.get("source_timestamp", ""),
            "source_date": str(row.get("source_timestamp", "")).split("T")[0],
            "source_time": str(row.get("source_timestamp", "")).split("T")[-1],
            "split_set": "test",
            "label": row.get("label", ""),
            "is_defective": row.get("is_defective", ""),
            "n_images": row.get("n_images", ""),
            "source_classes": row.get("source_classes", ""),
            "view_pairs": row.get("view_pairs", ""),
            "has_mask": "True" if is_defective(row) else "False",
            "image_ids": row.get("image_ids", ""),
            "relative_paths": row.get("relative_paths", ""),
            "raw_dataset_id": RAW_DATASET_ID,
            "manifest_id": GATE_ID,
            "piece_event_id": source_event_id(row),
            "dataset_version": GATE_ID,
            "replay_id": "",
            "validation_id": GATE_ID,
            "scenario_version": SCENARIO_VERSION,
            "validation_set_id": GATE_ID,
            "validation_role": GATE_ROLE,
        }
    )
    return mapped


def to_train_replay_row(row: dict[str, str], sequence_number: int) -> dict[str, str]:
    mapped = row.copy()
    mapped.update(
        {
            "scenario_id": TRAIN_SCENARIO_ID,
            "sequence_number": str(sequence_number),
            "manifest_id": TRAIN_MANIFEST_ID,
            "dataset_version": TRAIN_DATASET_VERSION,
            "replay_id": TRAIN_DATASET_VERSION,
            "scenario_version": TRAIN_MANIFEST_ID,
            "piece_event_id": mapped.get("simulated_event_id", ""),
        }
    )
    return mapped


def build_split(
    *,
    replay_path: Path = SOURCE_REPLAY,
    train_output: Path = TRAIN_REPLAY_OUTPUT,
    gate_output: Path = GATE_OUTPUT,
    good_total: int = HOLDOUT_GOOD_TOTAL,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    replay_rows = read_csv_rows(replay_path)
    holdout_rows = select_holdout_rows(replay_rows, good_total=good_total)
    holdout_ids = {source_event_id(row) for row in holdout_rows}

    train_rows = [
        to_train_replay_row(row, sequence_number=index)
        for index, row in enumerate((row for row in replay_rows if source_event_id(row) not in holdout_ids), start=1)
    ]
    gate_rows = [to_gate_row(row) for row in holdout_rows]

    replay_fieldnames = list(contract_for_key("natural_replay_train_v4").required_columns)
    gate_fieldnames = list(contract_for_key("validation_gate_set_v3").required_columns)
    write_csv_rows(train_output, train_rows, replay_fieldnames)
    write_csv_rows(gate_output, gate_rows, gate_fieldnames)
    return train_rows, gate_rows


def print_summary(train_rows: list[dict[str, str]], gate_rows: list[dict[str, str]]) -> None:
    for name, rows in [("train_replay", train_rows), ("gate_holdout", gate_rows)]:
        labels = Counter("defective" if is_defective(row) else "good" for row in rows)
        classes = Counter((source_class(row), "defective" if is_defective(row) else "good") for row in rows)
        print(f"{name}: rows={len(rows)} labels={dict(sorted(labels.items()))}")
        print(f"{name}: source_class_label={dict(sorted(classes.items()))}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, default=SOURCE_REPLAY)
    parser.add_argument("--train-output", type=Path, default=TRAIN_REPLAY_OUTPUT)
    parser.add_argument("--gate-output", type=Path, default=GATE_OUTPUT)
    parser.add_argument("--good-total", type=int, default=HOLDOUT_GOOD_TOTAL)
    args = parser.parse_args(argv)
    train_rows, gate_rows = build_split(
        replay_path=args.replay,
        train_output=args.train_output,
        gate_output=args.gate_output,
        good_total=args.good_total,
    )
    print(f"Wrote {args.train_output}")
    print(f"Wrote {args.gate_output}")
    print_summary(train_rows, gate_rows)


if __name__ == "__main__":
    main()
