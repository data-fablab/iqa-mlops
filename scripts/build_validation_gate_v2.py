"""Build a fixed, piece-level Feature-AE promotion gate panel v002."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from iqa.metadata.contracts import RAW_DATASET_ID, contract_for_key


DEFAULT_SOURCE = Path("data/metadata/casting_piece_events.csv")
DEFAULT_REPLAY = Path("data/metadata/casting_flux_replay_plan_natural_v003.csv")
DEFAULT_BOOTSTRAP = Path("data/metadata/feature_ae_bootstrap_events.csv")
DEFAULT_CALIBRATION = Path("data/validation/calibration_good_reference_v001.csv")
DEFAULT_REPRESENTATIVE = Path("data/validation/validation_set_replay_representative_v001.csv")
DEFAULT_GATE_V1 = Path("data/validation/validation_set_replay_gate_v001.csv")
DEFAULT_OUTPUT = Path("data/validation/validation_set_replay_gate_v002.csv")

GATE_ID = "validation_set_replay_gate_v002"
GATE_ROLE = "mvp_gate_piece_level_reference"
SCENARIO_VERSION = "feature_ae_mvp_v002"
GOOD_TARGET_TOTAL = 120


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(contract_for_key("validation_gate_set_v2").required_columns)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fieldnames} for row in rows])


def is_defective(row: dict[str, str]) -> bool:
    return str(row.get("is_defective") or "").strip().lower() == "true"


def source_key(row: dict[str, str]) -> tuple[str, str]:
    return str(row.get("source_class") or ""), str(row.get("group_key") or row.get("source_group_key") or "")


def source_event_id(row: dict[str, str]) -> str:
    return str(row.get("event_id") or row.get("source_event_id") or "")


def natural_replay_good_class_counts(replay_rows: list[dict[str, str]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in replay_rows:
        if not is_defective(row):
            counts[str(row.get("source_class") or "")] += 1
    return counts


def target_good_counts(replay_rows: list[dict[str, str]], total: int) -> dict[str, int]:
    counts = natural_replay_good_class_counts(replay_rows)
    observed_total = sum(counts.values())
    if observed_total <= 0:
        return {}
    raw = {source_class: total * count / observed_total for source_class, count in counts.items()}
    targets = {source_class: int(value) for source_class, value in raw.items()}
    remainder = total - sum(targets.values())
    ranked = sorted(raw, key=lambda source_class: raw[source_class] - targets[source_class], reverse=True)
    for source_class in ranked[:remainder]:
        targets[source_class] += 1
    return targets


def event_ids(rows: list[dict[str, str]], key: str = "event_id") -> set[str]:
    return {str(row.get(key) or "") for row in rows if row.get(key)}


def select_good_rows(
    source_rows: list[dict[str, str]],
    replay_rows: list[dict[str, str]],
    excluded_ids: set[str],
    *,
    total: int,
) -> list[dict[str, str]]:
    targets = target_good_counts(replay_rows, total)
    candidates_by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        if is_defective(row) or source_event_id(row) in excluded_ids:
            continue
        candidates_by_class[str(row.get("source_class") or "")].append(row)

    selected: list[dict[str, str]] = []
    for source_class, target in sorted(targets.items()):
        candidates = sorted(
            candidates_by_class[source_class],
            key=lambda row: (
                int(row.get("n_images") or 0) == 1,
                row.get("source_timestamp") or "",
                row.get("event_id") or "",
            ),
        )
        selected.extend(candidates[:target])

    return sorted(selected, key=lambda row: (row.get("source_class") or "", row.get("source_timestamp") or ""))


def select_defect_rows(source_rows: list[dict[str, str]], reserve_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    reserve_keys = {source_key(row) for row in reserve_rows if is_defective(row)}
    defects = [row for row in source_rows if is_defective(row) and source_key(row) in reserve_keys]
    return sorted(defects, key=lambda row: (row.get("source_class") or "", row.get("source_timestamp") or ""))


def normalize_validation_row(row: dict[str, str]) -> dict[str, str]:
    normalized = {field: row.get(field, "") for field in contract_for_key("validation_gate_set_v2").required_columns}
    normalized.update(
        {
            "raw_dataset_id": RAW_DATASET_ID,
            "manifest_id": GATE_ID,
            "piece_event_id": row.get("event_id", ""),
            "dataset_version": GATE_ID,
            "replay_id": "",
            "validation_id": GATE_ID,
            "scenario_version": SCENARIO_VERSION,
            "validation_set_id": GATE_ID,
            "validation_role": GATE_ROLE,
        }
    )
    return normalized


def build_gate_v2(
    *,
    source_path: Path = DEFAULT_SOURCE,
    replay_path: Path = DEFAULT_REPLAY,
    bootstrap_path: Path = DEFAULT_BOOTSTRAP,
    calibration_path: Path = DEFAULT_CALIBRATION,
    representative_path: Path = DEFAULT_REPRESENTATIVE,
    gate_v1_path: Path = DEFAULT_GATE_V1,
    output_path: Path = DEFAULT_OUTPUT,
    good_total: int = GOOD_TARGET_TOTAL,
) -> list[dict[str, str]]:
    source_rows = read_csv_rows(source_path)
    replay_rows = read_csv_rows(replay_path)
    bootstrap_rows = read_csv_rows(bootstrap_path)
    calibration_rows = read_csv_rows(calibration_path)
    representative_rows = read_csv_rows(representative_path)
    gate_v1_rows = read_csv_rows(gate_v1_path)

    replay_source_ids = event_ids(replay_rows, "source_event_id")
    excluded_good_ids = replay_source_ids | event_ids(bootstrap_rows) | event_ids(calibration_rows) | event_ids(representative_rows)
    reserve_rows = [*representative_rows, *gate_v1_rows]

    defect_rows = select_defect_rows(source_rows, reserve_rows)
    good_rows = select_good_rows(source_rows, replay_rows, excluded_good_ids, total=good_total)
    rows = [normalize_validation_row(row) for row in [*defect_rows, *good_rows]]
    rows = sorted(rows, key=lambda row: (row["is_defective"].lower() != "true", row["source_class"], row["source_timestamp"]))
    write_csv_rows(output_path, rows)
    return rows


def print_summary(rows: list[dict[str, str]]) -> None:
    counts = Counter("defective" if is_defective(row) else "good" for row in rows)
    by_class = Counter((row["source_class"], "defective" if is_defective(row) else "good") for row in rows)
    n_images = Counter(row["n_images"] for row in rows)
    print(f"rows: {len(rows)}")
    print(f"labels: {dict(sorted(counts.items()))}")
    print(f"source_class/label: {dict(sorted(by_class.items()))}")
    print(f"n_images: {dict(sorted(n_images.items()))}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--good-total", type=int, default=GOOD_TARGET_TOTAL)
    args = parser.parse_args(argv)
    rows = build_gate_v2(output_path=args.output, good_total=args.good_total)
    print(f"Wrote {args.output}")
    print_summary(rows)


if __name__ == "__main__":
    main()
