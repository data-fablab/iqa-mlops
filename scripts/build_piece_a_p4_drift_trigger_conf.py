"""Build smoke-test Airflow trigger confs for Piece B -> Piece A/P4 drift.

The generated conf targets ``iqa_lifecycle_trigger`` and injects synthetic
window metrics. It is useful for validating the trigger contract, but the
natural scenario uses ``iqa_drift_piece_a_p4`` plus
``iqa-run-drift-observation-replay`` so detection comes from replay inference.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal


SCENARIO_ID = "production_replay_natural_piece_b_to_piece_a_p4_drift"
DEFAULT_PLAN = Path("data/metadata/casting_flux_replay_plan_piece_b_to_piece_a_p4_drift_v001.csv")
DEFAULT_OUTPUT_ROOT = Path(".cache/iqa/drift_triggers")
STABLE_CLASSIFICATION_MODEL = "feature_ae_classifier__production_replay_natural_piece_b_full"
STABLE_LOCALIZATION_MODEL = "feature_ae_localization__production_replay_natural_piece_b_full"
PIECE_A_P4_VIEW_PAIR = "Casting_class1:2_3"
Phase = Literal["clear", "suspected", "confirmed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["clear", "suspected", "confirmed"], required=True)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--candidate-init-policy", choices=["active"], default="active")
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--lifecycle-interval", type=int, default=50)
    parser.add_argument("--ml-image", default="iqa-ml:local")
    parser.add_argument("--image-root", default="/opt/iqa/iqa-mlops/.cache/iqa/source_datasets/hss-iad")
    return parser.parse_args()


def read_plan(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"drift replay plan not found: {path}")
    with path.open(newline="", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row.get("scenario_id") == SCENARIO_ID]
    if not rows:
        raise ValueError(f"plan has no rows for scenario_id={SCENARIO_ID}: {path}")
    return rows


def validate_plan(rows: list[dict[str, str]]) -> dict[str, Any]:
    phase_counts = Counter(row.get("scenario_phase") or "" for row in rows)
    required_phases = {
        "stable_baseline_piece_b",
        "drift_piece_a_p4_suspected",
        "drift_piece_a_p4_confirmed",
        "correction_replay",
    }
    missing = sorted(phase for phase in required_phases if phase_counts.get(phase, 0) <= 0)
    if missing:
        raise ValueError(f"drift plan is missing required phases: {missing}")

    p4_rows = [row for row in rows if row.get("scenario_phase") != "stable_baseline_piece_b"]
    bad_source = sorted({row.get("source_class") or "" for row in p4_rows if row.get("source_class") != "Casting_class1"})
    bad_view_pairs = sorted({row.get("view_pairs") or "" for row in p4_rows if row.get("view_pairs") != PIECE_A_P4_VIEW_PAIR})
    if bad_source:
        raise ValueError(f"P4 drift rows must stay in Casting_class1, got: {bad_source}")
    if bad_view_pairs:
        raise ValueError(f"P4 drift rows must use {PIECE_A_P4_VIEW_PAIR}, got: {bad_view_pairs}")

    return {
        "row_count": len(rows),
        "phase_counts": dict(sorted(phase_counts.items())),
        "p4_event_count": len(p4_rows),
        "p4_label_counts": dict(sorted(Counter(row.get("label") or "" for row in p4_rows).items())),
    }


def _window_conf(phase: Phase, rows: list[dict[str, str]]) -> dict[str, Any]:
    phase_counts = Counter(row.get("scenario_phase") or "" for row in rows)
    suspected_count = phase_counts["drift_piece_a_p4_suspected"]
    confirmed_count = phase_counts["drift_piece_a_p4_confirmed"]

    if phase == "clear":
        return {
            "window_events": min(60, phase_counts["stable_baseline_piece_b"]),
            "domain_ratio": 0.0,
            "alert_rate": 0.02,
            "red_rate": 0.0,
            "unexpected_red_rate": 0.0,
            "roi_fail_rate": 0.0,
            "oracle_fn_rate": 0.0,
            "critical_window_count": 0,
            "expected_monitoring_status": "clear",
            "expected_trigger_lifecycle": False,
            "window_source": "stable_baseline_piece_b",
        }
    if phase == "suspected":
        return {
            "window_events": suspected_count,
            "domain_ratio": 1.0,
            "alert_rate": 0.20,
            "red_rate": 0.05,
            "unexpected_red_rate": 0.20,
            "roi_fail_rate": 0.0,
            "oracle_fn_rate": 0.0,
            "critical_window_count": 0,
            "expected_monitoring_status": "suspected",
            "expected_trigger_lifecycle": False,
            "window_source": "drift_piece_a_p4_suspected",
        }
    return {
        "window_events": suspected_count + confirmed_count,
        "domain_ratio": 1.0,
        "alert_rate": 0.55,
        "red_rate": 0.15,
        "unexpected_red_rate": 0.55,
        "roi_fail_rate": 0.0,
        "oracle_fn_rate": 0.0,
        "critical_window_count": 1,
        "expected_monitoring_status": "confirmed",
        "expected_trigger_lifecycle": True,
        "window_source": "drift_piece_a_p4_suspected+drift_piece_a_p4_confirmed",
    }


def build_conf(*, phase: Phase, rows: list[dict[str, str]], epochs: int, max_events: int | None, lifecycle_interval: int, ml_image: str, image_root: str) -> dict[str, Any]:
    plan_summary = validate_plan(rows)
    window = _window_conf(phase, rows)
    return {
        "scenario_id": SCENARIO_ID,
        "conforming_validated_count": 0,
        "drift_confirmed": False,
        "roi_fail_rate": window["roi_fail_rate"],
        "source_domain": "piece_a_p4",
        "window_events": window["window_events"],
        "domain_ratio": window["domain_ratio"],
        "alert_rate": window["alert_rate"],
        "red_rate": window["red_rate"],
        "unexpected_red_rate": window["unexpected_red_rate"],
        "oracle_fn_rate": window["oracle_fn_rate"],
        "critical_window_count": window["critical_window_count"],
        "api_url": "http://iqa-api:8000",
        "thresholds_config": "configs/monitoring_thresholds.yaml",
        "ml_image": ml_image,
        "repo_root": "/opt/iqa/iqa-mlops",
        "image_root": image_root,
        "mode": "progressive-train",
        "max_events": max_events or plan_summary["row_count"],
        "lifecycle_interval": lifecycle_interval,
        "max_cycles": None,
        "epochs": epochs,
        "max_steps": None,
        "gate_eval_profile": "full",
        "target_stage": "test",
        "promotion_min_delta": 0.0,
        "dual_promotion": True,
        "localization_promotion_min_delta": 0.0,
        "classification_require_fn_improvement": True,
        "classification_min_image_recall_delta": 0.0,
        "classification_min_image_ap_delta": 0.0,
        "anchor_good_manifest": "data/metadata/feature_ae_bootstrap_piece_b_minimal_v001.csv",
        "anchor_good_max_per_class": 256,
        "reference_eval_manifest": "data/validation/validation_set_piece_b_to_piece_a_p4_drift_v001.csv",
        "classification_selection_manifest": "data/validation/classification_selection_piece_b_to_piece_a_p4_drift_v001.csv",
        "reference_gt_masks_manifest": "data/validation/validation_gt_masks_piece_b_to_piece_a_p4_drift_v001.csv",
        "max_good_red_regression": 1,
        "candidate_init_policy": "active",
        "initial_classification_registered_model": STABLE_CLASSIFICATION_MODEL,
        "initial_localization_registered_model": STABLE_LOCALIZATION_MODEL,
        "require_mlflow_registry": True,
        "mlflow_tracking_uri": "http://mlflow:5000",
        "mlflow_s3_endpoint_url": "http://minio:9000",
        "s3_endpoint_url": "http://minio:9000",
        "scenario_validation": {
            "phase": phase,
            "window_source": window["window_source"],
            "expected_monitoring_status": window["expected_monitoring_status"],
            "expected_trigger_lifecycle": window["expected_trigger_lifecycle"],
            "plan_summary": plan_summary,
        },
    }


def main() -> None:
    args = parse_args()
    rows = read_plan(args.plan)
    phase = args.phase
    conf = build_conf(
        phase=phase,
        rows=rows,
        epochs=args.epochs,
        max_events=args.max_events,
        lifecycle_interval=args.lifecycle_interval,
        ml_image=args.ml_image,
        image_root=args.image_root,
    )
    output = args.output or DEFAULT_OUTPUT_ROOT / f"iqa_lifecycle_trigger_piece_a_p4_{phase}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(conf, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "written", "phase": phase, "output": str(output), "scenario_validation": conf["scenario_validation"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
