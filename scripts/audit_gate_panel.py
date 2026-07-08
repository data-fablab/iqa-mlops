"""Audit validation gate panel representativeness against a replay plan."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DatasetSummary:
    name: str
    rows: list[dict[str, str]]
    group_field: str

    @property
    def group_count(self) -> int:
        return len({row.get(self.group_field, "") for row in self.rows})


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def label(row: dict[str, str]) -> str:
    raw = str(row.get("label") or "").strip().lower()
    if raw:
        return raw
    return "defective" if as_bool(row.get("is_defective")) else "good"


def source_class(row: dict[str, str]) -> str:
    return str(row.get("source_class") or "-")


def group_key(row: dict[str, str], group_field: str) -> str:
    return str(row.get(group_field) or "-")


def n_images(row: dict[str, str]) -> int:
    try:
        return int(float(str(row.get("n_images") or "1")))
    except ValueError:
        return 1


def timestamp(row: dict[str, str]) -> str:
    return str(row.get("source_timestamp") or row.get("event_time") or row.get("scheduled_at") or "")


def count_by(rows: Iterable[dict[str, str]], *fields: str, group_field: str | None = None) -> Counter[tuple[str, ...]]:
    counts: Counter[tuple[str, ...]] = Counter()
    seen: set[str] = set()
    for row in rows:
        if group_field is not None:
            key = group_key(row, group_field)
            if key in seen:
                continue
            seen.add(key)
        values = []
        for field in fields:
            if field == "label":
                values.append(label(row))
            elif field == "source_class":
                values.append(source_class(row))
            else:
                values.append(str(row.get(field) or "-"))
        counts[tuple(values)] += 1
    return counts


def defect_group_keys(summary: DatasetSummary) -> set[tuple[str, str]]:
    return {
        (source_class(row), group_key(row, summary.group_field))
        for row in summary.rows
        if label(row) == "defective" or as_bool(row.get("is_defective"))
    }


def group_keys(summary: DatasetSummary) -> set[str]:
    return {group_key(row, summary.group_field) for row in summary.rows}


def time_range(rows: Iterable[dict[str, str]]) -> tuple[str, str] | tuple[None, None]:
    values = sorted(value for row in rows if (value := timestamp(row)))
    if not values:
        return None, None
    return values[0], values[-1]


def n_images_distribution(rows: Iterable[dict[str, str]]) -> Counter[int]:
    return Counter(n_images(row) for row in rows)


def print_counter(title: str, counts: Counter[tuple[str, ...]]) -> None:
    print(f"\n== {title} ==")
    if not counts:
        print("(empty)")
        return
    for key, value in sorted(counts.items()):
        print(f"{' / '.join(key)}: {value}")


def summarize_dataset(summary: DatasetSummary) -> None:
    print(f"\n# {summary.name}")
    print(f"rows: {len(summary.rows)}")
    print(f"groups: {summary.group_count}")
    print(f"time_range: {time_range(summary.rows)}")
    print_counter("rows by label", count_by(summary.rows, "label"))
    print_counter("rows by source_class / label", count_by(summary.rows, "source_class", "label"))
    print_counter(
        "groups by source_class / label",
        count_by(summary.rows, "source_class", "label", group_field=summary.group_field),
    )
    print("\n== n_images distribution ==")
    for images, value in sorted(n_images_distribution(summary.rows).items()):
        print(f"{images}: {value}")


def warning_lines(
    gate: DatasetSummary,
    replay: DatasetSummary,
    calibration: DatasetSummary | None,
    *,
    expect_disjoint_holdout: bool = False,
) -> list[str]:
    warnings: list[str] = []
    gate_labels = count_by(gate.rows, "label")
    replay_labels = count_by(replay.rows, "label")
    gate_good_rows = gate_labels.get(("good",), 0)
    replay_good_rows = replay_labels.get(("good",), 0)
    gate_defect_rows = gate_labels.get(("defective",), 0)
    replay_defect_rows = replay_labels.get(("defective",), 0)

    if gate_good_rows < 50:
        warnings.append(
            f"Gate has only {gate_good_rows} good rows; p95/p99 threshold calibration is unstable."
        )
    if gate.group_count < 30:
        warnings.append(f"Gate has only {gate.group_count} distinct groups; it is a small decision panel.")
    if gate_good_rows and gate_defect_rows:
        gate_defect_rate = gate_defect_rows / (gate_good_rows + gate_defect_rows)
        replay_defect_rate = replay_defect_rows / (replay_good_rows + replay_defect_rows)
        if abs(gate_defect_rate - replay_defect_rate) > 0.20:
            warnings.append(
                f"Gate defect rate {gate_defect_rate:.1%} is far from replay defect rate {replay_defect_rate:.1%}."
            )

    gate_images = n_images_distribution(gate.rows)
    replay_images = n_images_distribution(replay.rows)
    if set(gate_images) == {1} and any(images > 1 for images in replay_images):
        warnings.append("Gate is image-level only, while replay contains multi-image piece events.")

    gate_defects = defect_group_keys(gate)
    replay_defects = defect_group_keys(replay)
    missing_defects = replay_defects - gate_defects
    if not expect_disjoint_holdout and replay_defects and len(missing_defects) / len(replay_defects) > 0.5:
        warnings.append(
            f"Gate misses {len(missing_defects)}/{len(replay_defects)} replay defect source_class/group combinations."
        )

    overlap = group_keys(gate) & group_keys(replay)
    if not expect_disjoint_holdout and len(overlap) < max(3, min(len(group_keys(gate)), len(group_keys(replay))) * 0.2):
        warnings.append(f"Low group overlap between gate and replay: {len(overlap)} shared groups.")

    if calibration is not None:
        cal_good = count_by(calibration.rows, "label").get(("good",), 0)
        if cal_good > gate_good_rows:
            warnings.append(
                f"Calibration good reference has {cal_good} good rows vs gate {gate_good_rows}; "
                "consider using it for stable threshold calibration."
            )
    return warnings


def audit(
    gate_path: Path,
    replay_path: Path,
    calibration_path: Path | None = None,
    *,
    expect_disjoint_holdout: bool = False,
) -> None:
    gate = DatasetSummary("gate", read_csv_rows(gate_path), "group_key")
    replay = DatasetSummary("replay", read_csv_rows(replay_path), "source_group_key")
    calibration = (
        DatasetSummary("calibration", read_csv_rows(calibration_path), "group_key")
        if calibration_path is not None
        else None
    )

    print(f"gate_path: {gate_path}")
    print(f"replay_path: {replay_path}")
    if calibration_path is not None:
        print(f"calibration_path: {calibration_path}")

    summarize_dataset(gate)
    summarize_dataset(replay)
    if calibration is not None:
        summarize_dataset(calibration)

    gate_defects = defect_group_keys(gate)
    replay_defects = defect_group_keys(replay)
    missing = sorted(replay_defects - gate_defects)
    print("\n# coverage")
    print(f"shared_groups: {len(group_keys(gate) & group_keys(replay))}")
    print(f"gate_defect_class_groups: {len(gate_defects)}")
    print(f"replay_defect_class_groups: {len(replay_defects)}")
    if expect_disjoint_holdout:
        print("mode: disjoint_holdout")
        print("gate and replay are expected to be disjoint; compare distributions, not exact overlap.")
    else:
        print(f"missing_replay_defect_class_groups: {len(missing)}")
        for source, group in missing[:30]:
            print(f"missing_defect: {source} / {group}")
        if len(missing) > 30:
            print(f"... {len(missing) - 30} more")

    warnings = warning_lines(gate, replay, calibration, expect_disjoint_holdout=expect_disjoint_holdout)
    print("\n# warnings")
    if not warnings:
        print("(none)")
    for item in warnings:
        print(f"- {item}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, default=Path("data/validation/validation_set_replay_gate_v003.csv"))
    parser.add_argument(
        "--replay",
        type=Path,
        default=Path("data/metadata/casting_flux_replay_plan_natural_train_v004.csv"),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("data/validation/calibration_good_reference_v001.csv"),
    )
    parser.add_argument(
        "--expect-disjoint-holdout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat gate and replay as a deliberate disjoint split.",
    )
    args = parser.parse_args()
    audit(
        args.gate,
        args.replay,
        args.calibration if args.calibration.exists() else None,
        expect_disjoint_holdout=args.expect_disjoint_holdout,
    )


if __name__ == "__main__":
    main()
