"""Run a replay-driven IQA lifecycle simulation from real manifests."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from iqa.inference.feature_ae import predict_feature_ae_image
from iqa.inference.segmentation import predict_roi_image
from iqa.models.artifacts import (
    DEFAULT_FEATURE_AE_MODEL_VERSION,
    DEFAULT_ROI_MODEL_VERSION,
    load_feature_ae_decision_thresholds,
    resolve_feature_ae_checkpoint,
    resolve_roi_segmenter_checkpoint,
)
from iqa.monitoring import LifecycleDecision, LifecycleSignal, evaluate_lifecycle_signal
from iqa.runtime import gpu_lock
from iqa.storage.object_store import ObjectStore
from iqa.storage.visual_artifacts import (
    VisualArtifactContext,
    create_visual_object_store,
    publish_heatmap,
    publish_roi_mask,
)
from iqa.training.bootstrap import upload_checkpoint_to_s3
from iqa.training.feature_ae import FeatureAETrainingConfig
from iqa.training.feature_ae_contracts import FEATURE_AE_BUSINESS_METRIC_PRIORITY
from iqa.training.mlflow_logging import train_feature_ae_with_mlflow_logging

NATURAL_SCENARIO_ID = "production_replay_natural"
DRIFT_SCENARIO_ID = "drift_domain_extension"
REPLAY_PLANS = {
    NATURAL_SCENARIO_ID: Path("data/metadata/casting_flux_replay_plan_natural.csv"),
    DRIFT_SCENARIO_ID: Path("data/metadata/casting_flux_replay_plan_drift.csv"),
}
CANDIDATE_DATASETS = {
    NATURAL_SCENARIO_ID: "feature_ae_good_v002",
    DRIFT_SCENARIO_ID: "feature_ae_good_v003",
}
VALIDATION_MANIFEST = Path("data/validation/validation_set_v001.csv")
DEFAULT_OUTPUT_ROOT = Path(".cache/iqa/replay_lifecycle")
Mode = Literal["decision-only", "train-on-trigger", "progressive-decision", "progressive-train"]
PROGRESSIVE_MODES = {"progressive-decision", "progressive-train"}
ACTIVE_REPLAY_SCENARIOS = Path("data/metadata/replay_scenarios.csv")


@dataclass
class CycleEvent:
    event_id: str
    piece_event_id: str
    lot_id: str
    scenario_id: str
    source_class: str
    dataset_version: str
    relative_path: str
    image_path: str
    oracle_verdict: str
    decision: str
    score: float
    roi_quality_status: str
    roi_ratio: float
    threshold_orange: float
    threshold_red: float
    threshold_source: str
    roi_mask_path: str
    roi_mask_uri: str | None
    heatmap_path: str
    heatmap_uri: str | None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class LotAccumulator:
    lot_id: str
    scenario_id: str
    source_classes: set[str] = field(default_factory=set)
    dataset_versions: set[str] = field(default_factory=set)
    event_count: int = 0
    conforming_validated_count: int = 0
    defective_count: int = 0
    roi_fail_count: int = 0
    decisions: dict[str, int] = field(default_factory=dict)

    def add(self, event: CycleEvent) -> None:
        self.event_count += 1
        self.source_classes.add(event.source_class)
        self.dataset_versions.add(event.dataset_version)
        self.decisions[event.decision] = self.decisions.get(event.decision, 0) + 1
        if event.oracle_verdict == "conforme":
            self.conforming_validated_count += 1
        else:
            self.defective_count += 1
        if event.roi_quality_status == "fail":
            self.roi_fail_count += 1

    @property
    def roi_fail_rate(self) -> float:
        return self.roi_fail_count / self.event_count if self.event_count else 0.0

    def to_dict(self, *, lifecycle_decision: LifecycleDecision | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "lot_id": self.lot_id,
            "scenario_id": self.scenario_id,
            "event_count": self.event_count,
            "conforming_validated_count": self.conforming_validated_count,
            "defective_count": self.defective_count,
            "roi_fail_count": self.roi_fail_count,
            "roi_fail_rate": self.roi_fail_rate,
            "decisions": self.decisions,
            "source_classes": sorted(self.source_classes),
            "dataset_versions": sorted(self.dataset_versions),
        }
        if lifecycle_decision is not None:
            payload["lifecycle_decision"] = lifecycle_decision.to_dict()
            payload["trigger_lifecycle"] = lifecycle_decision.trigger_lifecycle
            payload["trigger_reason"] = lifecycle_decision.trigger_reason
            payload["candidate_dataset_version"] = lifecycle_decision.candidate_dataset_version
        return payload


@dataclass
class CycleState:
    scenario_id: str
    mode: Mode
    run_id: str
    output_dir: Path
    events_processed: int = 0
    lots_processed: int = 0
    total_conforming_validated_count: int = 0
    last_cycle_conforming_validated_count: int = 0
    trigger_decision: LifecycleDecision | None = None
    candidate_checkpoint: str | None = None
    mlflow_run_id: str | None = None
    status: str = "validated"
    active_model_initial: str = DEFAULT_FEATURE_AE_MODEL_VERSION
    active_model_final: str = DEFAULT_FEATURE_AE_MODEL_VERSION
    cycles_requested: int = 0
    cycles: list[dict[str, Any]] = field(default_factory=list)
    promotion_chain: list[str] = field(default_factory=lambda: [DEFAULT_FEATURE_AE_MODEL_VERSION])
    seen_events: list[CycleEvent] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        payload = {
            "scenario_id": self.scenario_id,
            "mode": self.mode,
            "run_id": self.run_id,
            "events_processed": self.events_processed,
            "lots_processed": self.lots_processed,
            "trigger_lifecycle": bool(self.trigger_decision and self.trigger_decision.trigger_lifecycle),
            "trigger_reason": self.trigger_decision.trigger_reason if self.trigger_decision else "",
            "candidate_dataset_version": self.trigger_decision.candidate_dataset_version if self.trigger_decision else "",
            "bootstrap_model_version": DEFAULT_FEATURE_AE_MODEL_VERSION,
            "candidate_checkpoint": self.candidate_checkpoint,
            "mlflow_run_id": self.mlflow_run_id,
            "status": self.status,
            "output_dir": str(self.output_dir),
        }
        if self.mode in PROGRESSIVE_MODES:
            payload.update(
                {
                    "active_model_initial": self.active_model_initial,
                    "active_model_final": self.active_model_final,
                    "cycles_requested": self.cycles_requested,
                    "cycles_completed": len(self.cycles),
                    "models_promoted": [
                        str(cycle["candidate_version"])
                        for cycle in self.cycles
                        if cycle.get("promotion_status") == "promoted"
                    ],
                    "promotion_chain": self.promotion_chain,
                    "registry_stage": self.cycles[-1]["registry_stage"] if self.cycles else "",
                    "metric_history": [
                        {
                            "cycle_id": cycle.get("cycle_id"),
                            "candidate_version": cycle.get("candidate_version"),
                            "selected_metric": cycle.get("selected_metric"),
                            "selected_metric_value": cycle.get("selected_metric_value"),
                            "gate_decision": cycle.get("gate_decision"),
                            "promotion_status": cycle.get("promotion_status"),
                        }
                        for cycle in self.cycles
                    ],
                    "best_cycle": _best_cycle_id(self.cycles),
                    "best_metric": _best_metric_name(self.cycles),
                    "best_metric_value": _best_metric_value(self.cycles),
                    "rejected_candidates": [
                        str(cycle["candidate_version"])
                        for cycle in self.cycles
                        if cycle.get("promotion_status") == "rejected"
                    ],
                }
            )
        return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-id", choices=sorted(REPLAY_PLANS), default=NATURAL_SCENARIO_ID)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--stage", default="test")
    parser.add_argument(
        "--mode",
        choices=["decision-only", "train-on-trigger", "progressive-decision", "progressive-train"],
        default="decision-only",
    )
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--max-lots", type=int)
    parser.add_argument("--publish-minio", action="store_true")
    parser.add_argument("--wait-for-gpu", action="store_true")
    parser.add_argument("--no-gpu-lock", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--lifecycle-interval", type=int, default=50)
    parser.add_argument("--max-cycles", type=int)
    parser.add_argument("--target-stage", default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_gpu_lock:
        result = run_cycle(args)
    else:
        with gpu_lock(owner="iqa-replay-lifecycle", blocking=args.wait_for_gpu):
            result = run_cycle(args)
    print(json.dumps(result, indent=2, sort_keys=True))


def run_cycle(args: argparse.Namespace) -> dict[str, Any]:
    state = CycleState(
        scenario_id=args.scenario_id,
        mode=args.mode,
        run_id=f"replay_lifecycle_{uuid4().hex}",
        output_dir=args.output_root / args.scenario_id,
        cycles_requested=args.max_cycles or 0,
    )
    state.output_dir = state.output_dir / state.run_id
    state.output_dir.mkdir(parents=True, exist_ok=True)

    roi_checkpoint = resolve_roi_segmenter_checkpoint(DEFAULT_ROI_MODEL_VERSION, strict_checksum=True)
    feature_checkpoint = resolve_feature_ae_checkpoint(DEFAULT_FEATURE_AE_MODEL_VERSION, strict_checksum=True)
    decision_thresholds = resolve_runtime_thresholds(DEFAULT_FEATURE_AE_MODEL_VERSION)
    visual_store = create_visual_object_store()
    rows = load_replay_rows(args.scenario_id)
    events_path = state.output_dir / "events.jsonl"
    lots_path = state.output_dir / "lots.jsonl"
    cycles_path = state.output_dir / "cycles.jsonl"

    current_lot: LotAccumulator | None = None
    with events_path.open("w", encoding="utf-8") as events_file, lots_path.open("w", encoding="utf-8") as lots_file:
        for row in rows:
            if args.max_events is not None and state.events_processed >= args.max_events:
                break
            lot_id = row.get("lot_id") or "unknown_lot"
            if current_lot is not None and current_lot.lot_id != lot_id:
                decision = _finalize_lot(current_lot, args=args, state=state, lots_file=lots_file)
                if handle_lifecycle_decision(args, state, decision):
                    break
                if args.mode == "progressive-train" and state.candidate_checkpoint:
                    feature_checkpoint = Path(state.candidate_checkpoint)
                if args.max_lots is not None and state.lots_processed >= args.max_lots:
                    break
            if current_lot is None or current_lot.lot_id != lot_id:
                current_lot = LotAccumulator(lot_id=lot_id, scenario_id=args.scenario_id)

            event = process_replay_event(
                row,
                image_root=args.image_root,
                roi_checkpoint=roi_checkpoint,
                feature_checkpoint=feature_checkpoint,
                decision_thresholds=decision_thresholds,
                output_dir=state.output_dir,
                device=args.device,
                visual_store=visual_store,
            )
            current_lot.add(event)
            state.seen_events.append(event)
            state.events_processed += 1
            events_file.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")

        if current_lot is not None and should_finalize_last_lot(args, state):
            decision = _finalize_lot(current_lot, args=args, state=state, lots_file=lots_file)
            handle_lifecycle_decision(args, state, decision)
            if args.mode == "progressive-train" and state.candidate_checkpoint:
                feature_checkpoint = Path(state.candidate_checkpoint)

    if state.trigger_decision and state.trigger_decision.trigger_lifecycle and args.mode == "train-on-trigger":
        train_result = train_candidate_on_trigger(args, state.trigger_decision)
        state.candidate_checkpoint = str(train_result.get("checkpoint") or "")
        state.mlflow_run_id = str(train_result.get("run_id") or "")
        state.status = "trained"
        if args.publish_minio and state.candidate_checkpoint:
            upload_checkpoint_to_s3(
                state.candidate_checkpoint,
                f"s3://iqa-models/{state.trigger_decision.candidate_dataset_version}/checkpoint.pt",
            )

    if state.cycles:
        cycles_path.write_text(
            "".join(json.dumps(cycle, sort_keys=True) + "\n" for cycle in state.cycles),
            encoding="utf-8",
        )

    summary = state.summary()
    (state.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def load_replay_rows(scenario_id: str) -> list[dict[str, str]]:
    plan = resolve_replay_plan(scenario_id)
    with plan.open(newline="", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row.get("scenario_id") == scenario_id]
    if not rows:
        raise ValueError(f"replay plan has no rows for scenario_id={scenario_id}: {plan}")
    return rows


def resolve_replay_plan(scenario_id: str) -> Path:
    if ACTIVE_REPLAY_SCENARIOS.exists():
        with ACTIVE_REPLAY_SCENARIOS.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row.get("scenario_id") == scenario_id and row.get("output_path"):
                    configured = Path(row["output_path"])
                    if configured.exists():
                        return configured
                    repo_relative = Path("data/metadata") / configured.name
                    if repo_relative.exists():
                        return repo_relative
    return REPLAY_PLANS[scenario_id]


def process_replay_event(
    row: dict[str, str],
    *,
    image_root: Path,
    roi_checkpoint: Path,
    feature_checkpoint: Path,
    decision_thresholds: dict[str, Any],
    output_dir: Path,
    device: str,
    visual_store: ObjectStore | None = None,
) -> CycleEvent:
    relative_path = first_csv_value(row.get("relative_paths") or row.get("relative_path") or "")
    image_path = image_root / relative_path
    image_id = first_csv_value(row.get("image_ids") or row.get("image_id") or Path(relative_path).stem)
    event_id = row.get("event_id") or row.get("simulated_event_id") or ""
    piece_event_id = row.get("piece_event_id") or row.get("simulated_event_id") or row.get("event_id") or ""
    lot_id = row.get("lot_id") or "unknown_lot"
    scenario_id = row.get("scenario_id") or ""
    mask_path = output_dir / "roi_masks" / f"{piece_event_id}_{image_id}_roi.png"
    heatmap_path = output_dir / "heatmaps" / f"{piece_event_id}_{image_id}_heatmap.png"
    context = VisualArtifactContext(
        scenario_id=scenario_id,
        lot_id=lot_id,
        piece_event_id=piece_event_id,
        image_id=image_id,
    )
    roi = predict_roi_image(image_path, roi_checkpoint, device=device, output_mask=mask_path)
    roi_mask_uri = publish_roi_mask(mask_path, context, store=visual_store) if mask_path.exists() else None
    feature = predict_feature_ae_image(
        image_path,
        feature_checkpoint,
        device=device,
        roi_mask_path=mask_path,
        heatmap_output_path=heatmap_path,
        threshold_orange=float(decision_thresholds["threshold_orange"]),
        threshold_red=float(decision_thresholds["threshold_red"]),
        threshold_source=str(decision_thresholds["threshold_source"]),
    )
    heatmap_uri = publish_heatmap(heatmap_path, context, store=visual_store) if heatmap_path.exists() else None
    return CycleEvent(
        event_id=event_id,
        piece_event_id=piece_event_id,
        lot_id=lot_id,
        scenario_id=scenario_id,
        source_class=row.get("source_class") or "",
        dataset_version=row.get("dataset_version") or "",
        relative_path=relative_path,
        image_path=str(image_path),
        oracle_verdict=oracle_verdict(row),
        decision=feature.status,
        score=feature.score,
        roi_quality_status=roi.roi_quality_status,
        roi_ratio=roi.roi_ratio,
        threshold_orange=feature.threshold_orange,
        threshold_red=feature.threshold_red,
        threshold_source=feature.threshold_source,
        roi_mask_path=str(mask_path),
        roi_mask_uri=roi_mask_uri,
        heatmap_path=str(heatmap_path),
        heatmap_uri=heatmap_uri,
    )


def resolve_runtime_thresholds(model_version: str) -> dict[str, Any]:
    thresholds = load_feature_ae_decision_thresholds(model_version)
    if thresholds:
        return {
            "threshold_orange": float(thresholds["threshold_orange"]),
            "threshold_red": float(thresholds["threshold_red"]),
            "threshold_source": f"manifest:{thresholds.get('method', 'decision_thresholds')}",
        }
    return {
        "threshold_orange": 0.02,
        "threshold_red": 0.05,
        "threshold_source": "legacy_default",
    }


def first_csv_value(value: str) -> str:
    return value.split("|", 1)[0].split(";", 1)[0].split(",", 1)[0].strip()


def oracle_verdict(row: dict[str, str]) -> str:
    is_defective = str(row.get("is_defective") or row.get("oracle_is_defective") or "").lower()
    has_mask = str(row.get("has_mask") or "").lower()
    if is_defective in {"true", "1", "yes", "defective"} or has_mask in {"true", "1", "yes"}:
        return "defective"
    return "conforme"


def lifecycle_decision_for_lot(state: CycleState, lot: LotAccumulator, *, interval: int = 50) -> LifecycleDecision:
    state.total_conforming_validated_count += lot.conforming_validated_count
    conforming_since_cycle = state.total_conforming_validated_count - state.last_cycle_conforming_validated_count
    signal = LifecycleSignal(
        scenario_id=state.scenario_id,
        conforming_validated_count=conforming_since_cycle,
        drift_confirmed=state.scenario_id == DRIFT_SCENARIO_ID,
        roi_fail_rate=lot.roi_fail_rate,
    )
    return evaluate_lifecycle_signal(signal, min_natural_conforming=interval)


def _finalize_lot(
    lot: LotAccumulator,
    *,
    args: argparse.Namespace,
    state: CycleState,
    lots_file: Any,
) -> LifecycleDecision:
    decision = lifecycle_decision_for_lot(state, lot, interval=args.lifecycle_interval)
    state.lots_processed += 1
    lots_file.write(json.dumps(lot.to_dict(lifecycle_decision=decision), sort_keys=True) + "\n")
    return decision


def should_finalize_last_lot(args: argparse.Namespace, state: CycleState) -> bool:
    if state.trigger_decision is None:
        return True
    return args.mode in PROGRESSIVE_MODES and not reached_max_cycles(args, state)


def handle_lifecycle_decision(args: argparse.Namespace, state: CycleState, decision: LifecycleDecision) -> bool:
    if not decision.trigger_lifecycle:
        return False
    state.trigger_decision = decision
    if args.mode not in PROGRESSIVE_MODES:
        return True

    cycle_number = len(state.cycles) + 1
    cycle_result = build_progressive_cycle(args, state, decision, cycle_number)
    state.cycles.append(cycle_result)
    state.last_cycle_conforming_validated_count = state.total_conforming_validated_count
    if cycle_result.get("promotion_status") == "promoted":
        promoted = str(cycle_result["candidate_version"])
        state.active_model_final = promoted
        state.promotion_chain.append(promoted)
        state.candidate_checkpoint = str(cycle_result.get("candidate_checkpoint") or "")
        state.mlflow_run_id = str(cycle_result.get("mlflow_run_id") or "")
        state.status = "trained" if args.mode == "progressive-train" else "validated"
    elif args.mode == "progressive-train":
        state.status = "rejected"
    return reached_max_cycles(args, state)


def reached_max_cycles(args: argparse.Namespace, state: CycleState) -> bool:
    return args.max_cycles is not None and len(state.cycles) >= args.max_cycles


def build_progressive_cycle(
    args: argparse.Namespace,
    state: CycleState,
    decision: LifecycleDecision,
    cycle_number: int,
) -> dict[str, Any]:
    candidate_version = f"rd_feature_ae_gated_natural_cycle_{cycle_number:03d}"
    dataset_snapshot_id = f"feature_ae_natural_cycle_{cycle_number:03d}"
    calibration_set_id = f"calibration_natural_cycle_{cycle_number:03d}"
    cycle_dir = state.output_dir / "cycles" / f"cycle_{cycle_number:03d}"
    cycle_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = write_seen_dataset_snapshot(
        state.seen_events,
        cycle_dir / f"{dataset_snapshot_id}.csv",
        dataset_snapshot_id=dataset_snapshot_id,
        scenario_id=state.scenario_id,
    )
    result: dict[str, Any] = {
        "cycle_id": f"cycle_{cycle_number:03d}",
        "candidate_version": candidate_version,
        "dataset_snapshot_id": dataset_snapshot_id,
        "dataset_snapshot_path": str(manifest_path),
        "calibration_set_id": calibration_set_id,
        "seen_events": len(state.seen_events),
        "seen_conforming": sum(1 for event in state.seen_events if event.oracle_verdict == "conforme"),
        "seen_defective": sum(1 for event in state.seen_events if event.oracle_verdict == "defective"),
        "trigger_reason": decision.trigger_reason,
        "registry_stage": args.target_stage,
        "promotion_status": "simulated",
        "gate_decision": "not_run",
        "gate_reason": "decision_only",
        "metrics": {},
        "selected_metric": None,
        "selected_metric_value": None,
        "selected_epoch": None,
        "selected_checkpoint": None,
        "val_loss": None,
    }
    if args.mode == "progressive-train":
        train_result = train_progressive_candidate(args, candidate_version, manifest_path, dataset_snapshot_id)
        result["candidate_checkpoint"] = str(train_result.get("checkpoint") or "")
        result["mlflow_run_id"] = str(train_result.get("run_id") or "")
        result.update(metric_evidence_from_training_result(train_result))
        if result["selected_metric"]:
            result["promotion_status"] = "promoted"
            result["gate_decision"] = "passed"
            result["gate_reason"] = "business_metric_available"
        else:
            result["promotion_status"] = "rejected"
            result["gate_decision"] = "rejected"
            result["gate_reason"] = "missing_business_metric"
        if args.publish_minio and result["candidate_checkpoint"] and result["promotion_status"] == "promoted":
            upload_checkpoint_to_s3(
                str(result["candidate_checkpoint"]),
                f"s3://iqa-models/{candidate_version}/checkpoint.pt",
            )
    else:
        result["candidate_checkpoint"] = None
        result["mlflow_run_id"] = None
    return result


def metric_evidence_from_training_result(train_result: dict[str, Any]) -> dict[str, Any]:
    run_dir_value = train_result.get("run_dir")
    checkpoint_value = train_result.get("checkpoint")
    candidates: list[Path] = []
    if run_dir_value:
        run_dir = Path(str(run_dir_value))
        candidates.extend([run_dir / "metric_eval_best.json", run_dir / "bootstrap_run" / "metric_eval_best.json"])
    if checkpoint_value:
        checkpoint_dir = Path(str(checkpoint_value)).parent
        candidates.extend([checkpoint_dir / "metric_eval_best.json", checkpoint_dir / "bootstrap_run" / "metric_eval_best.json"])

    best_path = next((path for path in candidates if path.is_file()), None)
    best = json.loads(best_path.read_text(encoding="utf-8")) if best_path else {}
    metrics = {
        metric: float(record["value"])
        for metric, record in best.items()
        if isinstance(record, dict)
        and record.get("value") is not None
        and metric in {*FEATURE_AE_BUSINESS_METRIC_PRIORITY, "pixel_auroc"}
    }
    selected_metric = None
    selected_record: dict[str, Any] | None = None
    for metric in FEATURE_AE_BUSINESS_METRIC_PRIORITY:
        record = best.get(metric)
        if isinstance(record, dict) and record.get("value") is not None:
            selected_metric = metric
            selected_record = record
            break

    val_loss = None
    if selected_record and run_dir_value:
        val_loss = _val_loss_for_epoch(Path(str(run_dir_value)) / "loss_history.csv", int(selected_record.get("epoch") or 0))

    return {
        "metrics": metrics,
        "metric_eval_best_path": str(best_path) if best_path else None,
        "selected_metric": selected_metric,
        "selected_metric_value": float(selected_record["value"]) if selected_record else None,
        "selected_epoch": int(selected_record.get("epoch") or 0) if selected_record else None,
        "selected_checkpoint": str(selected_record.get("checkpoint") or "") if selected_record else None,
        "val_loss": val_loss,
    }


def _val_loss_for_epoch(path: Path, epoch: int) -> float | None:
    if not path.is_file() or epoch <= 0:
        return None
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if int(row.get("epoch") or 0) == epoch and row.get("val_loss"):
                return float(row["val_loss"])
    return None


def _metric_sort_key(cycle: dict[str, Any]) -> tuple[int, float]:
    metric = cycle.get("selected_metric")
    if metric not in FEATURE_AE_BUSINESS_METRIC_PRIORITY:
        return (-1, float("-inf"))
    return (
        len(FEATURE_AE_BUSINESS_METRIC_PRIORITY) - FEATURE_AE_BUSINESS_METRIC_PRIORITY.index(str(metric)),
        float(cycle.get("selected_metric_value") or float("-inf")),
    )


def _best_cycle(cycles: list[dict[str, Any]]) -> dict[str, Any] | None:
    promoted = [cycle for cycle in cycles if cycle.get("promotion_status") == "promoted"]
    return max(promoted, key=_metric_sort_key) if promoted else None


def _best_cycle_id(cycles: list[dict[str, Any]]) -> str | None:
    cycle = _best_cycle(cycles)
    return str(cycle.get("cycle_id")) if cycle else None


def _best_metric_name(cycles: list[dict[str, Any]]) -> str | None:
    cycle = _best_cycle(cycles)
    return str(cycle.get("selected_metric")) if cycle else None


def _best_metric_value(cycles: list[dict[str, Any]]) -> float | None:
    cycle = _best_cycle(cycles)
    return float(cycle["selected_metric_value"]) if cycle and cycle.get("selected_metric_value") is not None else None


def write_seen_dataset_snapshot(
    events: list[CycleEvent],
    output_path: Path,
    *,
    dataset_snapshot_id: str,
    scenario_id: str,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "image_ids",
        "relative_path",
        "relative_paths",
        "event_id",
        "source_class",
        "split_set",
        "label",
        "is_defective",
        "scenario_id",
        "dataset_version",
        "manifest_version",
        "gt_mask_path",
        "oracle_verdict",
        "train_eligible",
        "train_eligibility_source",
        "quarantine_reason",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            if event.oracle_verdict != "conforme":
                continue
            image_id = Path(event.relative_path).stem
            writer.writerow(
                {
                    "image_id": image_id,
                    "image_ids": image_id,
                    "relative_path": event.relative_path,
                    "relative_paths": event.relative_path,
                    "event_id": event.event_id,
                    "source_class": event.source_class,
                    "split_set": scenario_id,
                    "label": "good",
                    "is_defective": "False",
                    "scenario_id": scenario_id,
                    "dataset_version": dataset_snapshot_id,
                    "manifest_version": f"{dataset_snapshot_id}_manifest_v001",
                    "gt_mask_path": "",
                    "oracle_verdict": "conforme",
                    "train_eligible": "true",
                    "train_eligibility_source": "oracle_gt_seen_lots",
                    "quarantine_reason": "",
                }
            )
    return output_path


def train_progressive_candidate(
    args: argparse.Namespace,
    candidate_version: str,
    manifest_path: Path,
    dataset_snapshot_id: str,
) -> dict[str, Any]:
    run_dir = Path(".cache/iqa/models") / candidate_version
    run_dir.mkdir(parents=True, exist_ok=True)
    config = FeatureAETrainingConfig(
        manifest_path=manifest_path,
        image_root=args.image_root,
        output_checkpoint=run_dir / "checkpoint.pt",
        scenario_id=args.scenario_id,
        dataset_version=dataset_snapshot_id,
        manifest_version=f"{dataset_snapshot_id}_manifest_v001",
        candidate_version=candidate_version,
        roi_model_version=DEFAULT_ROI_MODEL_VERSION,
        feature_ae_version=DEFAULT_FEATURE_AE_MODEL_VERSION,
        run_name=f"{candidate_version}_{args.target_stage}",
        device=args.device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_steps=args.max_steps,
        metric_eval_manifest_path=VALIDATION_MANIFEST,
        metric_eval_device=args.device,
        metric_eval_every_epochs=1,
        metric_eval_start_epoch=1,
        metric_eval_calibrate_normal=True,
        metric_eval_apply_score_region_to_map=True,
        require_business_metric_for_early_stopping=True,
    )
    return train_feature_ae_with_mlflow_logging(config, git_commit=_git_commit())


def train_candidate_on_trigger(args: argparse.Namespace, decision: LifecycleDecision) -> dict[str, Any]:
    candidate_version = decision.candidate_dataset_version
    if not candidate_version:
        raise ValueError("lifecycle decision did not provide a candidate_dataset_version")
    manifest_path = Path("data/model_datasets") / f"{candidate_version}.csv"
    run_dir = Path(".cache/iqa/models") / candidate_version
    run_dir.mkdir(parents=True, exist_ok=True)
    config = FeatureAETrainingConfig(
        manifest_path=manifest_path,
        image_root=args.image_root,
        output_checkpoint=run_dir / "checkpoint.pt",
        scenario_id=args.scenario_id,
        dataset_version=candidate_version,
        manifest_version=f"{candidate_version}_manifest_v001",
        candidate_version=candidate_version,
        roi_model_version=DEFAULT_ROI_MODEL_VERSION,
        feature_ae_version=DEFAULT_FEATURE_AE_MODEL_VERSION,
        run_name=f"{candidate_version}_{args.stage}",
        device=args.device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_steps=args.max_steps,
        metric_eval_manifest_path=VALIDATION_MANIFEST,
        metric_eval_device=args.device,
        metric_eval_every_epochs=1,
        metric_eval_start_epoch=1,
        metric_eval_calibrate_normal=True,
        metric_eval_apply_score_region_to_map=True,
        require_business_metric_for_early_stopping=True,
    )
    return train_feature_ae_with_mlflow_logging(config, git_commit=_git_commit())


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
