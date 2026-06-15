"""Finalize IQA Phase 1 data contracts from Casting manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EVENTS_PATH = Path("data/metadata/casting_piece_events.csv")
BOOTSTRAP_PATH = Path("data/metadata/feature_ae_bootstrap_events.csv")
VALIDATION_PATH = Path("data/validation/validation_set_v001.csv")
CALIBRATION_PATH = Path("data/metadata/calibration_set_v001.csv")
NATURAL_REPLAY_PATH = Path("data/metadata/casting_flux_replay_plan_natural.csv")
DRIFT_REPLAY_PATH = Path("data/metadata/casting_flux_replay_plan_drift.csv")
REPORT_PATH = Path("reports/data_phase1_validation.md")

VALIDATION_DEFECTIVE_BY_CLASS = {
    "Casting_class1": 3,
    "Casting_class2": 6,
    "Casting_class3": 5,
}
VALIDATION_GOOD_BY_CLASS = 2
CALIBRATION_GOOD_BY_CLASS = 20
RECORDED_AT = "2026-06-12T00:00:00"
ROI_MODEL_VERSION = "roi_segmenter_v001_fixed"
FEATURE_AE_VERSION = "rd_feature_ae_gated_v001_bootstrap"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=EVENTS_PATH)
    parser.add_argument("--bootstrap", type=Path, default=BOOTSTRAP_PATH)
    parser.add_argument("--validation-output", type=Path, default=VALIDATION_PATH)
    parser.add_argument("--calibration-output", type=Path, default=CALIBRATION_PATH)
    parser.add_argument("--natural-replay", type=Path, default=NATURAL_REPLAY_PATH)
    parser.add_argument("--drift-replay", type=Path, default=DRIFT_REPLAY_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    return parser.parse_args()


def _sort_events(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["source_class", "source_timestamp", "event_id"], kind="stable")


def build_validation_set(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    test_events = _sort_events(events[events["split_set"].eq("test")])
    for source_class in sorted(test_events["source_class"].unique()):
        class_events = test_events[test_events["source_class"].eq(source_class)]
        rows.append(class_events[class_events["label"].eq("good")].head(VALIDATION_GOOD_BY_CLASS))
        defective_count = VALIDATION_DEFECTIVE_BY_CLASS[source_class]
        rows.append(class_events[class_events["is_defective"].eq(True)].head(defective_count))
    validation = pd.concat(rows, ignore_index=True)
    validation["validation_set_id"] = "validation_set_v001"
    validation["validation_role"] = "metric_best_selection"
    return validation


def build_calibration_set(
    events: pd.DataFrame,
    *,
    bootstrap_event_ids: set[str],
    validation_event_ids: set[str],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    candidates = _sort_events(
        events[
            events["split_set"].eq("train")
            & events["label"].eq("good")
            & events["is_defective"].eq(False)
            & ~events["event_id"].isin(bootstrap_event_ids | validation_event_ids)
        ]
    )
    for source_class in sorted(candidates["source_class"].unique()):
        rows.append(candidates[candidates["source_class"].eq(source_class)].head(CALIBRATION_GOOD_BY_CLASS))
    calibration = pd.concat(rows, ignore_index=True)
    calibration["calibration_set_id"] = "calibration_set_v001"
    calibration["calibration_role"] = "feature_ae_threshold_calibration"
    return calibration


def update_replay_plan(path: Path, *, excluded_source_event_ids: set[str]) -> pd.DataFrame:
    plan = pd.read_csv(path)
    plan = plan[~plan["source_event_id"].isin(excluded_source_event_ids)].copy()
    plan["sequence_number"] = range(1, len(plan) + 1)
    plan["event_time"] = plan["scheduled_at"]
    plan["recorded_at"] = RECORDED_AT
    plan["is_simulated"] = True
    plan["dataset_version"] = plan["scenario_id"].map(
        {
            "production_replay_natural": "production_replay_natural_v001",
            "drift_domain_extension": "drift_domain_extension_v001",
        }
    )
    plan["roi_model_version"] = ROI_MODEL_VERSION
    plan["feature_ae_version"] = FEATURE_AE_VERSION
    plan.to_csv(path, index=False)
    return plan


def _write_report(
    path: Path,
    *,
    events: pd.DataFrame,
    bootstrap: pd.DataFrame,
    validation: pd.DataFrame,
    calibration: pd.DataFrame,
    natural: pd.DataFrame,
    drift: pd.DataFrame,
    overlaps: dict[str, int],
) -> None:
    lines = [
        "# IQA Phase 1 Data Validation",
        "",
        "## Volumes",
        "",
        f"- piece_events: {len(events)}",
        f"- defective piece_events: {int(events['is_defective'].sum())}",
        f"- images referenced by piece_events: {int(events['n_images'].sum())}",
        f"- bootstrap events: {len(bootstrap)}",
        f"- validation_set_v001 events: {len(validation)}",
        f"- calibration_set_v001 events: {len(calibration)}",
        f"- production_replay_natural events: {len(natural)}",
        f"- drift_domain_extension events: {len(drift)}",
        "",
        "## Invariant",
        "",
        "`bootstrap ∩ calibration ∩ replay ∩ validation = empty`",
        "",
        *[f"- {name}: {count}" for name, count in overlaps.items()],
        "",
        "## Notes",
        "",
        "- `validation_set_v001` is frozen and used for metric-best selection.",
        "- `calibration_set_v001` is good-only and reserved for Feature-AE threshold calibration.",
        "- Replay plans keep defects for oracle feedback but exclude bootstrap, validation and calibration events.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    events = pd.read_csv(args.events)
    bootstrap = pd.read_csv(args.bootstrap)
    validation = build_validation_set(events)
    calibration = build_calibration_set(
        events,
        bootstrap_event_ids=set(bootstrap["event_id"]),
        validation_event_ids=set(validation["event_id"]),
    )

    args.validation_output.parent.mkdir(parents=True, exist_ok=True)
    args.calibration_output.parent.mkdir(parents=True, exist_ok=True)
    validation.to_csv(args.validation_output, index=False)
    calibration.to_csv(args.calibration_output, index=False)

    bootstrap_ids = set(bootstrap["event_id"])
    validation_ids = set(validation["event_id"])
    calibration_ids = set(calibration["event_id"])
    excluded_ids = bootstrap_ids | validation_ids | calibration_ids
    natural = update_replay_plan(args.natural_replay, excluded_source_event_ids=excluded_ids)
    drift = update_replay_plan(args.drift_replay, excluded_source_event_ids=excluded_ids)
    replay_ids = set(natural["source_event_id"]) | set(drift["source_event_id"])

    overlaps = {
        "bootstrap_vs_validation": len(bootstrap_ids & validation_ids),
        "bootstrap_vs_calibration": len(bootstrap_ids & calibration_ids),
        "bootstrap_vs_replay": len(bootstrap_ids & replay_ids),
        "validation_vs_calibration": len(validation_ids & calibration_ids),
        "validation_vs_replay": len(validation_ids & replay_ids),
        "calibration_vs_replay": len(calibration_ids & replay_ids),
    }
    if any(overlaps.values()):
        raise SystemExit(f"Phase 1 data invariant failed: {overlaps}")

    _write_report(
        args.report,
        events=events,
        bootstrap=bootstrap,
        validation=validation,
        calibration=calibration,
        natural=natural,
        drift=drift,
        overlaps=overlaps,
    )
    print(f"Wrote {args.validation_output}, {args.calibration_output} and {args.report}.")


if __name__ == "__main__":
    main()
