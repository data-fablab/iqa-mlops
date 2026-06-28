"""Run a replay-driven IQA lifecycle simulation from real manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
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
from iqa.dags.refresh_active_artifacts import refresh_active_artifacts
from iqa.monitoring import LifecycleDecision, LifecycleSignal, evaluate_lifecycle_signal
from iqa.monitoring.warmstart_resolver import resolve_warmstart_checkpoint
from iqa.runtime import gpu_lock
from iqa.storage.object_store import ObjectStore
from iqa.storage.artifacts import sha256_file
from iqa.storage.visual_artifacts import (
    VisualArtifactContext,
    create_visual_object_store,
    publish_heatmap,
    publish_roi_mask,
)
from iqa.training.bootstrap import upload_checkpoint_to_s3
from iqa.training.feature_ae import FeatureAETrainingConfig
from iqa.training.feature_ae_evaluation import (
    PREDICTION_SCHEMA_VERSION,
    FeatureAEEvaluationConfig,
    compute_decision_metrics,
    evaluate_feature_ae_checkpoint,
)
from iqa.training.feature_ae_contracts import CANONICAL_FEATURE_AE_PREPROCESSING, FEATURE_AE_BUSINESS_METRIC_PRIORITY
from iqa.training.mlflow_logging import train_feature_ae_with_mlflow_logging
from iqa.registry import register_run_to_model, registered_model_name

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
VALIDATION_GT_MASKS_MANIFEST = Path("data/validation/validation_gt_masks_v001.csv")
DEFAULT_ANCHOR_GOOD_MANIFEST = Path("data/model_datasets/feature_ae_good_v002.csv")
DEFAULT_OUTPUT_ROOT = Path(".cache/iqa/replay_lifecycle")
Mode = Literal["decision-only", "train-on-trigger", "progressive-decision", "progressive-train"]
PROGRESSIVE_MODES = {"progressive-decision", "progressive-train"}
ACTIVE_REPLAY_SCENARIOS = Path("data/metadata/replay_scenarios.csv")
PROGRESSIVE_PROMOTION_POLICY = "candidate_must_pass_reference_guardrail_and_progressive_factory_panel"


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
    roi_probability_path: str
    heatmap_path: str
    heatmap_uri: str | None
    gt_mask_path: str = ""
    active_model_version: str = DEFAULT_FEATURE_AE_MODEL_VERSION
    score_contract_version: str = ""

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
    promotion_status: str = ""
    triggering_class: str | None = None
    active_model_initial: str = DEFAULT_FEATURE_AE_MODEL_VERSION
    active_model_final: str = DEFAULT_FEATURE_AE_MODEL_VERSION
    cycles_requested: int = 0
    promotion_min_delta: float = 0.0
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
            "promotion_status": self.promotion_status,
            "triggering_class": self.triggering_class,
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
                    "registry_model_name": registered_model_name(self.scenario_id),
                    "promotion_policy": PROGRESSIVE_PROMOTION_POLICY,
                    "promotion_min_delta": getattr(self, "promotion_min_delta", 0.0),
                    "comparison_history": [
                        {
                            "cycle_id": cycle.get("cycle_id"),
                            "active_model_before": cycle.get("active_model_before"),
                            "candidate_version": cycle.get("candidate_version"),
                            "evaluation_seen_events": cycle.get("evaluation_seen_events"),
                            "selected_metric": cycle.get("selected_metric"),
                            "active_metric_value": cycle.get("active_metric_value"),
                            "candidate_metric_value": cycle.get("candidate_metric_value"),
                            "metric_delta": cycle.get("metric_delta"),
                            "gate_decision": cycle.get("gate_decision"),
                            "promotion_status": cycle.get("promotion_status"),
                            "registry_status": cycle.get("registry_status"),
                        }
                        for cycle in self.cycles
                    ],
                    "metric_history": [
                        {
                            "cycle_id": cycle.get("cycle_id"),
                            "candidate_version": cycle.get("candidate_version"),
                            "selected_metric": cycle.get("selected_metric"),
                            "selected_metric_value": cycle.get("selected_metric_value"),
                            "active_metric_value": cycle.get("active_metric_value"),
                            "candidate_metric_value": cycle.get("candidate_metric_value"),
                            "metric_delta": cycle.get("metric_delta"),
                            "gate_decision": cycle.get("gate_decision"),
                            "promotion_status": cycle.get("promotion_status"),
                        }
                        for cycle in self.cycles
                    ],
                    "best_cycle": _best_cycle_id(self.cycles),
                    "best_promoted_cycle": _best_cycle_id(self.cycles),
                    "best_candidate_seen": _best_candidate_seen(self.cycles),
                    "best_metric": _best_metric_name(self.cycles),
                    "best_metric_value": _best_metric_value(self.cycles),
                    "rejected_candidates": [
                        str(cycle["candidate_version"])
                        for cycle in self.cycles
                        if str(cycle.get("promotion_status", "")).startswith("rejected")
                    ],
                }
            )
        return payload


@dataclass
class ActiveRuntimeModel:
    version: str
    checkpoint: Path
    decision_thresholds: dict[str, Any]
    registry_model_name: str
    registry_stage: str
    registry_alias: str = ""
    registered_model_version: str = ""
    registry_status: str = ""
    registry_source_of_truth: str = ""
    score_contract_version: str = CANONICAL_FEATURE_AE_PREPROCESSING.version

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "checkpoint": str(self.checkpoint),
            "threshold_orange": self.decision_thresholds.get("threshold_orange"),
            "threshold_red": self.decision_thresholds.get("threshold_red"),
            "threshold_source": self.decision_thresholds.get("threshold_source"),
            "registry_model_name": self.registry_model_name,
            "registry_stage": self.registry_stage,
            "registry_alias": self.registry_alias,
            "registered_model_version": self.registered_model_version,
            "registry_status": self.registry_status,
            "registry_source_of_truth": self.registry_source_of_truth,
            "score_contract_version": self.score_contract_version,
        }


@dataclass
class LifecycleArtifacts:
    events_path: Path
    lots_path: Path
    cycles_path: Path
    summary_path: Path
    progress_path: Path
    lifecycle_events_path: Path
    timings_path: Path


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
    parser.add_argument("--promotion-min-delta", type=float, default=0.0)
    parser.add_argument("--require-mlflow-registry", action="store_true")
    parser.add_argument("--anchor-good-manifest", type=Path, default=DEFAULT_ANCHOR_GOOD_MANIFEST)
    parser.add_argument("--anchor-good-max-per-class", type=int, default=256)
    parser.add_argument("--reference-eval-manifest", type=Path, default=VALIDATION_MANIFEST)
    parser.add_argument("--reference-gt-masks-manifest", type=Path, default=VALIDATION_GT_MASKS_MANIFEST)
    parser.add_argument("--progressive-min-defects-for-decision", type=int, default=5)
    parser.add_argument("--max-good-red-regression", type=int, default=1)
    parser.add_argument("--candidate-init-policy", choices=["stable_base", "active", "fresh"], default="stable_base")
    parser.add_argument("--candidate-init-checkpoint", type=Path, default=None, help="explicit checkpoint path for warm-start (overrides --candidate-init-policy)")
    parser.add_argument("--triggering-class", default=None, help="class that triggered the drift (e.g. Casting_class2); selects the demo warm-start checkpoint and the new PatchCore coverage")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_gpu_lock:
        result = run_cycle(args)
    else:
        with gpu_lock(owner="iqa-replay-lifecycle", blocking=args.wait_for_gpu):
            result = run_cycle(args)
    print(json.dumps(result, indent=2, sort_keys=True))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, default=str, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, default=str, sort_keys=True) + "\n")
        file.flush()


def write_progress(
    artifacts: LifecycleArtifacts,
    state: CycleState,
    active_runtime: ActiveRuntimeModel,
    *,
    phase: str,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = state.summary()
    payload.update(
        {
            "phase": phase,
            "active_runtime_model": active_runtime.to_dict(),
            "active_model_version": active_runtime.version,
            "current_cycle": len(state.cycles),
            "last_cycle": state.cycles[-1] if state.cycles else None,
            "updated_at": datetime.now(UTC).isoformat(),
        }
    )
    if extra:
        payload.update(extra)
    write_json(artifacts.progress_path, payload)


def summary_with_runtime(
    state: CycleState,
    active_runtime: ActiveRuntimeModel,
    artifacts: LifecycleArtifacts,
) -> dict[str, Any]:
    payload = state.summary()
    payload.update(
        {
            "active_runtime_final": active_runtime.to_dict(),
            "progress_path": str(artifacts.progress_path),
            "lifecycle_events_path": str(artifacts.lifecycle_events_path),
            "cycles_path": str(artifacts.cycles_path),
            "timings_path": str(artifacts.timings_path),
        }
    )
    return payload


def record_lifecycle_event(
    artifacts: LifecycleArtifacts,
    state: CycleState,
    active_runtime: ActiveRuntimeModel,
    event_type: str,
    **payload: Any,
) -> None:
    append_jsonl(
        artifacts.lifecycle_events_path,
        {
            "event_type": event_type,
            "run_id": state.run_id,
            "scenario_id": state.scenario_id,
            "mode": state.mode,
            "events_processed": state.events_processed,
            "lots_processed": state.lots_processed,
            "active_model_version": active_runtime.version,
            "timestamp": datetime.now(UTC).isoformat(),
            **payload,
        },
    )


def record_timing(artifacts: LifecycleArtifacts, phase: str, *, duration_seconds: float, **payload: Any) -> None:
    append_jsonl(
        artifacts.timings_path,
        {
            "phase": phase,
            "duration_seconds": duration_seconds,
            "timestamp": datetime.now(UTC).isoformat(),
            **payload,
        },
    )


def run_cycle(args: argparse.Namespace) -> dict[str, Any]:
    state = CycleState(
        scenario_id=args.scenario_id,
        mode=args.mode,
        run_id=f"replay_lifecycle_{uuid4().hex}",
        output_dir=args.output_root / args.scenario_id,
        cycles_requested=args.max_cycles or 0,
        promotion_min_delta=float(args.promotion_min_delta),
    )
    state.output_dir = state.output_dir / state.run_id
    state.output_dir.mkdir(parents=True, exist_ok=True)

    roi_checkpoint = resolve_roi_segmenter_checkpoint(DEFAULT_ROI_MODEL_VERSION, strict_checksum=True)
    active_runtime = ActiveRuntimeModel(
        version=DEFAULT_FEATURE_AE_MODEL_VERSION,
        checkpoint=resolve_feature_ae_checkpoint(DEFAULT_FEATURE_AE_MODEL_VERSION, strict_checksum=True),
        decision_thresholds=resolve_runtime_thresholds(DEFAULT_FEATURE_AE_MODEL_VERSION),
        registry_model_name=registered_model_name(args.scenario_id),
        registry_stage=args.target_stage,
    )
    visual_store = create_visual_object_store()
    rows = load_replay_rows(args.scenario_id)
    events_path = state.output_dir / "events.jsonl"
    lots_path = state.output_dir / "lots.jsonl"
    cycles_path = state.output_dir / "cycles.jsonl"
    artifacts = LifecycleArtifacts(
        events_path=events_path,
        lots_path=lots_path,
        cycles_path=cycles_path,
        summary_path=state.output_dir / "summary.json",
        progress_path=state.output_dir / "progress.json",
        lifecycle_events_path=state.output_dir / "lifecycle_events.jsonl",
        timings_path=state.output_dir / "timings.jsonl",
    )
    write_progress(artifacts, state, active_runtime, phase="started")
    record_lifecycle_event(artifacts, state, active_runtime, "run_started", args=vars(args))

    current_lot: LotAccumulator | None = None
    with events_path.open("w", encoding="utf-8") as events_file, lots_path.open("w", encoding="utf-8") as lots_file:
        for row in rows:
            if args.max_events is not None and state.events_processed >= args.max_events:
                break
            lot_id = row.get("lot_id") or "unknown_lot"
            if current_lot is not None and current_lot.lot_id != lot_id:
                decision = _finalize_lot(current_lot, args=args, state=state, lots_file=lots_file)
                write_progress(artifacts, state, active_runtime, phase="lot_finalized", extra={"last_lot_id": current_lot.lot_id})
                should_stop, active_runtime = handle_lifecycle_decision(
                    args,
                    state,
                    decision,
                    active_runtime=active_runtime,
                    artifacts=artifacts,
                )
                if should_stop:
                    break
                if args.max_lots is not None and state.lots_processed >= args.max_lots:
                    break
            if current_lot is None or current_lot.lot_id != lot_id:
                current_lot = LotAccumulator(lot_id=lot_id, scenario_id=args.scenario_id)

            event = process_replay_event(
                row,
                image_root=args.image_root,
                roi_checkpoint=roi_checkpoint,
                feature_checkpoint=active_runtime.checkpoint,
                decision_thresholds=active_runtime.decision_thresholds,
                output_dir=state.output_dir,
                device=args.device,
                visual_store=visual_store,
                active_model_version=active_runtime.version,
            )
            current_lot.add(event)
            state.seen_events.append(event)
            state.events_processed += 1
            events_file.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
            events_file.flush()
            write_progress(artifacts, state, active_runtime, phase="replaying", extra={"current_lot_id": current_lot.lot_id})

        if current_lot is not None and should_finalize_last_lot(args, state):
            decision = _finalize_lot(current_lot, args=args, state=state, lots_file=lots_file)
            write_progress(artifacts, state, active_runtime, phase="lot_finalized", extra={"last_lot_id": current_lot.lot_id})
            _, active_runtime = handle_lifecycle_decision(
                args,
                state,
                decision,
                active_runtime=active_runtime,
                artifacts=artifacts,
            )

    if state.trigger_decision and state.trigger_decision.trigger_lifecycle and args.mode == "train-on-trigger":
        train_result = train_candidate_on_trigger(args, state.trigger_decision)
        state.candidate_checkpoint = str(train_result.get("checkpoint") or "")
        state.mlflow_run_id = str(train_result.get("run_id") or "")
        state.status = "trained"
        state.triggering_class = getattr(args, "triggering_class", None)
        if args.publish_minio and state.candidate_checkpoint:
            upload_checkpoint_to_s3(
                state.candidate_checkpoint,
                f"s3://iqa-models/{state.trigger_decision.candidate_dataset_version}/checkpoint.pt",
            )
        promote_refresh_reload_on_trigger(args, state, train_result)

    summary = summary_with_runtime(state, active_runtime, artifacts)
    write_json(artifacts.summary_path, summary)
    write_progress(artifacts, state, active_runtime, phase="completed")
    record_lifecycle_event(artifacts, state, active_runtime, "run_completed", summary=summary)
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
    active_model_version: str = DEFAULT_FEATURE_AE_MODEL_VERSION,
) -> CycleEvent:
    relative_path = first_csv_value(row.get("relative_paths") or row.get("relative_path") or "")
    image_path = image_root / relative_path
    image_id = first_csv_value(row.get("image_ids") or row.get("image_id") or Path(relative_path).stem)
    event_id = row.get("event_id") or row.get("simulated_event_id") or ""
    piece_event_id = row.get("piece_event_id") or row.get("simulated_event_id") or row.get("event_id") or ""
    lot_id = row.get("lot_id") or "unknown_lot"
    scenario_id = row.get("scenario_id") or ""
    mask_path = output_dir / "roi_masks" / f"{piece_event_id}_{image_id}_roi.png"
    probability_path = output_dir / "roi_masks" / f"{piece_event_id}_{image_id}_roi_prob.png"
    heatmap_path = output_dir / "heatmaps" / f"{piece_event_id}_{image_id}_heatmap.png"
    context = VisualArtifactContext(
        scenario_id=scenario_id,
        lot_id=lot_id,
        piece_event_id=piece_event_id,
        image_id=image_id,
    )
    roi = predict_roi_image(
        image_path,
        roi_checkpoint,
        device=device,
        output_mask=mask_path,
        output_probability_map=probability_path,
    )
    roi_mask_uri = publish_roi_mask(mask_path, context, store=visual_store) if mask_path.exists() else None
    feature = predict_feature_ae_image(
        image_path,
        feature_checkpoint,
        device=device,
        roi_mask_path=mask_path,
        roi_probability_path=probability_path,
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
        roi_probability_path=str(probability_path),
        gt_mask_path=resolve_event_gt_mask_path(row, relative_path),
        heatmap_path=str(heatmap_path),
        heatmap_uri=heatmap_uri,
        active_model_version=active_model_version,
        score_contract_version=getattr(feature, "score_contract_version", "feature_ae_reference_v001"),
    )


def resolve_runtime_thresholds(model_version: str) -> dict[str, Any]:
    thresholds = load_feature_ae_decision_thresholds(model_version)
    return {
        "threshold_orange": float(thresholds["threshold_orange"]),
        "threshold_red": float(thresholds["threshold_red"]),
        "threshold_source": f"manifest:{thresholds.get('method', 'decision_thresholds')}",
    }


def first_csv_value(value: str) -> str:
    return value.split("|", 1)[0].split(";", 1)[0].split(",", 1)[0].strip()


def gt_mask_path_for_original_dataset(relative_path: str) -> str:
    path = Path(relative_path)
    parts = path.parts
    if len(parts) < 4:
        return ""
    source_class, split, label = parts[0], parts[1].lower(), parts[2].lower()
    if split == "test" and label == "defective":
        return str(Path(source_class) / "ground_truth" / "defective" / f"{path.stem}_mask.png").replace("\\", "/")
    return ""


def resolve_event_gt_mask_path(row: dict[str, str], relative_path: str) -> str:
    explicit = first_csv_value(row.get("gt_mask_paths") or row.get("gt_mask_path") or row.get("mask_paths") or "")
    if explicit:
        return explicit
    return gt_mask_path_for_original_dataset(relative_path)


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
    lots_file.flush()
    return decision


def should_finalize_last_lot(args: argparse.Namespace, state: CycleState) -> bool:
    if state.trigger_decision is None:
        return True
    return args.mode in PROGRESSIVE_MODES and not reached_max_cycles(args, state)


def handle_lifecycle_decision(
    args: argparse.Namespace,
    state: CycleState,
    decision: LifecycleDecision,
    *,
    active_runtime: ActiveRuntimeModel,
    artifacts: LifecycleArtifacts,
) -> tuple[bool, ActiveRuntimeModel]:
    if not decision.trigger_lifecycle:
        return False, active_runtime
    state.trigger_decision = decision
    if args.mode not in PROGRESSIVE_MODES:
        return True, active_runtime

    cycle_number = len(state.cycles) + 1
    cycle_id = f"cycle_{cycle_number:03d}"
    record_lifecycle_event(
        artifacts,
        state,
        active_runtime,
        "cycle_started",
        cycle_id=cycle_id,
        trigger_reason=decision.trigger_reason,
    )
    write_progress(
        artifacts,
        state,
        active_runtime,
        phase="cycle_running",
        extra={"current_cycle": cycle_number, "trigger_reason": decision.trigger_reason},
    )
    cycle_result = build_progressive_cycle(
        args,
        state,
        decision,
        cycle_number,
        active_runtime=active_runtime,
        artifacts=artifacts,
    )
    state.last_cycle_conforming_validated_count = state.total_conforming_validated_count
    if cycle_result.get("mlflow_run_id"):
        record_lifecycle_event(
            artifacts,
            state,
            active_runtime,
            "candidate_trained",
            cycle_id=cycle_id,
            candidate_version=cycle_result.get("candidate_version"),
            candidate_checkpoint=cycle_result.get("candidate_checkpoint"),
            mlflow_run_id=cycle_result.get("mlflow_run_id"),
            mlflow_dataset_logged=cycle_result.get("mlflow_dataset_logged"),
            mlflow_model_logged=cycle_result.get("mlflow_model_logged"),
            selected_metric=cycle_result.get("selected_metric"),
            selected_metric_value=cycle_result.get("selected_metric_value"),
        )
    if cycle_result.get("evaluation_completed_at"):
        record_lifecycle_event(
            artifacts,
            state,
            active_runtime,
            "evaluation_completed",
            cycle_id=cycle_id,
            candidate_version=cycle_result.get("candidate_version"),
            selected_metric=cycle_result.get("selected_metric"),
            active_metric_value=cycle_result.get("active_metric_value"),
            candidate_metric_value=cycle_result.get("candidate_metric_value"),
            metric_delta=cycle_result.get("metric_delta"),
            evaluation_duration_seconds=cycle_result.get("evaluation_duration_seconds"),
            aupimo_stability=cycle_result.get("candidate_aupimo_stability"),
        )
    if cycle_result.get("promotion_status") == "promoted":
        activation_started = datetime.now(UTC)
        promoted = str(cycle_result["candidate_version"])
        state.active_model_final = promoted
        state.promotion_chain.append(promoted)
        state.candidate_checkpoint = str(cycle_result.get("candidate_checkpoint") or "")
        state.mlflow_run_id = str(cycle_result.get("mlflow_run_id") or "")
        state.status = "trained" if args.mode == "progressive-train" else "validated"
        active_runtime = ActiveRuntimeModel(
            version=promoted,
            checkpoint=Path(str(cycle_result["candidate_checkpoint"])),
            decision_thresholds=dict(cycle_result["candidate_decision_thresholds"]),
            registry_model_name=str(cycle_result.get("registered_model_name") or registered_model_name(state.scenario_id)),
            registry_stage=str(cycle_result.get("registry_stage") or args.target_stage),
            registry_alias=str(cycle_result.get("registry_alias") or args.target_stage),
            registered_model_version=str(cycle_result.get("registered_model_version") or ""),
            registry_status=str(cycle_result.get("registry_status") or ""),
            registry_source_of_truth=str(cycle_result.get("registry_source_of_truth") or ""),
        )
        cycle_result["activated_for_next_events"] = True
        cycle_result["activation_event_index"] = state.events_processed
        cycle_result["activation_scope"] = (
            "mlflow_registry" if cycle_result.get("registry_status") == "registered" else "run_local_runtime"
        )
        cycle_result["active_thresholds_after"] = active_runtime.decision_thresholds
        cycle_result["active_runtime_after"] = active_runtime.to_dict()
        record_lifecycle_event(
            artifacts,
            state,
            active_runtime,
            "gate_passed",
            cycle_id=cycle_id,
            candidate_version=promoted,
            selected_metric=cycle_result.get("selected_metric"),
            active_metric_value=cycle_result.get("active_metric_value"),
            candidate_metric_value=cycle_result.get("candidate_metric_value"),
            metric_delta=cycle_result.get("metric_delta"),
            promotion_status=cycle_result.get("promotion_status"),
            registry_status=cycle_result.get("registry_status"),
            aupimo_stability=cycle_result.get("candidate_aupimo_stability"),
        )
        if cycle_result.get("registry_status") == "failed":
            record_lifecycle_event(
                artifacts,
                state,
                active_runtime,
                "registry_failed",
                cycle_id=cycle_id,
                candidate_version=promoted,
                registry_reason=cycle_result.get("registry_reason"),
                activation_scope=cycle_result["activation_scope"],
            )
        record_lifecycle_event(
            artifacts,
            state,
            active_runtime,
            "model_activated",
            cycle_id=cycle_id,
            active_model_version=active_runtime.version,
            active_thresholds=active_runtime.decision_thresholds,
            activation_scope=cycle_result["activation_scope"],
        )
        record_timing(
            artifacts,
            "activation",
            duration_seconds=(datetime.now(UTC) - activation_started).total_seconds(),
            cycle_id=cycle_id,
            active_model_version=active_runtime.version,
        )
    elif args.mode == "progressive-train":
        state.status = "rejected"
        cycle_result["activated_for_next_events"] = False
        cycle_result["activation_event_index"] = state.events_processed
        cycle_result["activation_scope"] = "unchanged"
        cycle_result["active_thresholds_after"] = active_runtime.decision_thresholds
        cycle_result["active_runtime_after"] = active_runtime.to_dict()
        record_lifecycle_event(
            artifacts,
            state,
            active_runtime,
            "gate_rejected",
            cycle_id=cycle_id,
            candidate_version=cycle_result.get("candidate_version"),
            selected_metric=cycle_result.get("selected_metric"),
            active_metric_value=cycle_result.get("active_metric_value"),
            candidate_metric_value=cycle_result.get("candidate_metric_value"),
            metric_delta=cycle_result.get("metric_delta"),
            gate_reason=cycle_result.get("gate_reason"),
            promotion_status=cycle_result.get("promotion_status"),
            aupimo_stability=cycle_result.get("candidate_aupimo_stability"),
            operational_alerts=cycle_result.get("operational_alerts"),
        )
    else:
        cycle_result["activated_for_next_events"] = False
        cycle_result["activation_scope"] = "simulated"
        cycle_result["active_thresholds_after"] = active_runtime.decision_thresholds
        cycle_result["active_runtime_after"] = active_runtime.to_dict()
    state.cycles.append(cycle_result)
    append_jsonl(artifacts.cycles_path, cycle_result)
    write_json(artifacts.summary_path, summary_with_runtime(state, active_runtime, artifacts))
    write_progress(artifacts, state, active_runtime, phase="cycle_completed", extra={"last_cycle": cycle_result})
    print(
        "lifecycle cycle "
        f"{cycle_id}: gate={cycle_result.get('gate_decision')} "
        f"delta={cycle_result.get('metric_delta')} "
        f"promotion={cycle_result.get('promotion_status')} "
        f"active_model={active_runtime.version}",
        flush=True,
    )
    return reached_max_cycles(args, state), active_runtime


def reached_max_cycles(args: argparse.Namespace, state: CycleState) -> bool:
    return args.max_cycles is not None and len(state.cycles) >= args.max_cycles


def build_progressive_cycle(
    args: argparse.Namespace,
    state: CycleState,
    decision: LifecycleDecision,
    cycle_number: int,
    *,
    active_runtime: ActiveRuntimeModel,
    artifacts: LifecycleArtifacts,
) -> dict[str, Any]:
    candidate_version = f"rd_feature_ae_gated_natural_cycle_{cycle_number:03d}"
    dataset_snapshot_id = f"feature_ae_natural_cycle_{cycle_number:03d}"
    calibration_set_id = f"calibration_natural_cycle_{cycle_number:03d}"
    reference_evaluation_set_id = "reference_eval_v001"
    progressive_evaluation_set_id = f"progressive_eval_cycle_{cycle_number:03d}"
    active_model_before = active_runtime.version
    active_checkpoint = active_runtime.checkpoint
    cycle_dir = state.output_dir / "cycles" / f"cycle_{cycle_number:03d}"
    cycle_dir.mkdir(parents=True, exist_ok=True)
    seen_snapshot_path = write_seen_dataset_snapshot(
        state.seen_events,
        cycle_dir / f"{dataset_snapshot_id}.csv",
        dataset_snapshot_id=dataset_snapshot_id,
        scenario_id=state.scenario_id,
    )
    training_manifest_path, training_manifest_stats = write_progressive_training_manifest(
        seen_snapshot_path=seen_snapshot_path,
        anchor_good_manifest=args.anchor_good_manifest,
        output_path=cycle_dir / "training_manifest.csv",
        dataset_snapshot_id=dataset_snapshot_id,
        scenario_id=state.scenario_id,
        anchor_good_max_per_class=args.anchor_good_max_per_class,
    )
    progressive_evaluation_set_path = write_seen_evaluation_set(
        state.seen_events,
        cycle_dir / "progressive_evaluation_set.csv",
        evaluation_set_id=progressive_evaluation_set_id,
        scenario_id=state.scenario_id,
    )
    shutil.copy2(progressive_evaluation_set_path, cycle_dir / "evaluation_set.csv")
    reference_evaluation_set_path = cycle_dir / "reference_evaluation_set.csv"
    shutil.copy2(args.reference_eval_manifest, reference_evaluation_set_path)
    result: dict[str, Any] = {
        "cycle_id": f"cycle_{cycle_number:03d}",
        "promotion_policy": PROGRESSIVE_PROMOTION_POLICY,
        "promotion_min_delta": float(args.promotion_min_delta),
        "active_model_before": active_model_before,
        "active_runtime_before": active_runtime.to_dict(),
        "active_thresholds_before": active_runtime.decision_thresholds,
        "candidate_version": candidate_version,
        "dataset_snapshot_id": dataset_snapshot_id,
        "dataset_snapshot_path": str(seen_snapshot_path),
        "training_manifest_path": str(training_manifest_path),
        "training_manifest_stats": training_manifest_stats,
        "candidate_init_policy": args.candidate_init_policy,
        "calibration_set_id": calibration_set_id,
        "evaluation_set_id": progressive_evaluation_set_id,
        "evaluation_set_path": str(progressive_evaluation_set_path),
        "reference_evaluation_set_id": reference_evaluation_set_id,
        "reference_evaluation_set_path": str(reference_evaluation_set_path),
        "progressive_evaluation_set_id": progressive_evaluation_set_id,
        "progressive_evaluation_set_path": str(progressive_evaluation_set_path),
        "evaluation_seen_events": len(state.seen_events),
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
        "active_metrics_on_eval_set": {},
        "candidate_metrics_on_eval_set": {},
        "active_metric_value": None,
        "candidate_metric_value": None,
        "metric_delta": None,
        "registry_status": "not_registered",
        "candidate_decision_thresholds": None,
        "threshold_source": "",
        "active_thresholds_after": active_runtime.decision_thresholds,
        "operational_alerts": [],
    }
    if args.mode == "progressive-train":
        train_started = datetime.now(UTC)
        initial_checkpoint_path = resolve_candidate_initial_checkpoint(
            args,
            active_runtime=active_runtime,
        )
        train_result = train_progressive_candidate(
            args,
            candidate_version,
            training_manifest_path,
            dataset_snapshot_id,
            initial_checkpoint_path=initial_checkpoint_path,
        )
        train_completed = datetime.now(UTC)
        record_timing(
            artifacts,
            "train",
            duration_seconds=(train_completed - train_started).total_seconds(),
            cycle_id=f"cycle_{cycle_number:03d}",
            candidate_version=candidate_version,
        )
        result["candidate_checkpoint"] = str(train_result.get("checkpoint") or "")
        result["candidate_initial_checkpoint"] = str(initial_checkpoint_path) if initial_checkpoint_path else ""
        result["mlflow_run_id"] = str(train_result.get("run_id") or "")
        result["mlflow_dataset_logged"] = bool(train_result.get("mlflow_dataset_logged"))
        result["mlflow_training_dataset_logged"] = bool(train_result.get("mlflow_training_dataset_logged"))
        result["mlflow_metric_eval_dataset_logged"] = bool(train_result.get("mlflow_metric_eval_dataset_logged"))
        result["mlflow_model_logged"] = bool(train_result.get("mlflow_model_logged"))
        candidate_training_evidence = metric_evidence_from_training_result(train_result)
        result.update(candidate_training_evidence)
        result["epoch_selected_metric"] = result.get("selected_metric")
        result["epoch_selected_metric_value"] = result.get("selected_metric_value")
        result["epoch_selected_epoch"] = result.get("selected_epoch")
        result["epoch_selected_checkpoint"] = result.get("selected_checkpoint")
        write_progress(
            artifacts,
            state,
            active_runtime,
            phase="candidate_trained",
            extra={
                "current_cycle": cycle_number,
                "current_epoch": result.get("selected_epoch"),
                "best_epoch": result.get("selected_epoch"),
                "best_business_metric": result.get("selected_metric"),
                "best_business_metric_value": result.get("selected_metric_value"),
                "mlflow_run_id": result.get("mlflow_run_id"),
                "mlflow_dataset_logged": result.get("mlflow_dataset_logged"),
                "mlflow_model_logged": result.get("mlflow_model_logged"),
                "last_epoch_metrics": (result.get("epoch_metric_history") or [])[-1]
                if result.get("epoch_metric_history")
                else {},
            },
        )
        record_lifecycle_event(
            artifacts,
            state,
            active_runtime,
            "candidate_trained",
            cycle_id=f"cycle_{cycle_number:03d}",
            candidate_version=candidate_version,
            selected_epoch=result.get("selected_epoch"),
            selected_metric=result.get("selected_metric"),
            selected_metric_value=result.get("selected_metric_value"),
            mlflow_run_id=result.get("mlflow_run_id"),
            mlflow_dataset_logged=result.get("mlflow_dataset_logged"),
            mlflow_model_logged=result.get("mlflow_model_logged"),
        )
        comparison = evaluate_progressive_promotion_comparison(
            args,
            cycle_dir=cycle_dir,
            reference_evaluation_set_path=reference_evaluation_set_path,
            reference_evaluation_set_id=reference_evaluation_set_id,
            progressive_evaluation_set_path=progressive_evaluation_set_path,
            progressive_evaluation_set_id=progressive_evaluation_set_id,
            active_model_version=active_model_before,
            active_checkpoint_path=Path(active_checkpoint),
            candidate_version=candidate_version,
            candidate_checkpoint_path=Path(str(result["candidate_checkpoint"])),
            artifacts=artifacts,
            state=state,
            active_runtime=active_runtime,
        )
        result.update(comparison)
        if comparison["gate_decision"] == "passed":
            result["promotion_status"] = "promoted"
            result["gate_decision"] = "passed"
            result["gate_reason"] = comparison["gate_reason"]
        else:
            result["promotion_status"] = comparison["promotion_status"]
            result["gate_decision"] = "rejected"
            result["gate_reason"] = comparison["gate_reason"]
        if args.publish_minio and result["candidate_checkpoint"] and result["promotion_status"] == "promoted":
            upload_checkpoint_to_s3(
                str(result["candidate_checkpoint"]),
                f"s3://iqa-models/{candidate_version}/checkpoint.pt",
            )
        if result["promotion_status"] == "promoted":
            registry_started = datetime.now(UTC)
            result.update(register_promoted_cycle(args, state, result))
            registry_completed = datetime.now(UTC)
            record_timing(
                artifacts,
                "registry",
                duration_seconds=(registry_completed - registry_started).total_seconds(),
                cycle_id=f"cycle_{cycle_number:03d}",
                candidate_version=candidate_version,
                registry_status=result.get("registry_status"),
            )
        tag_mlflow_promotion_evidence(result)
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
    history_path = next((path.parent / "metric_eval_history.json" for path in candidates if (path.parent / "metric_eval_history.json").is_file()), None)
    epoch_metric_history = json.loads(history_path.read_text(encoding="utf-8")) if history_path else train_result.get("epoch_metric_history", [])
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
        "epoch_metric_history": epoch_metric_history,
        "checkpoint_selection_policy": train_result.get("checkpoint_selection_policy") or "business_metric_only",
    }


def evaluate_model_pair_on_panel(
    args: argparse.Namespace,
    *,
    cycle_dir: Path,
    panel_name: str,
    evaluation_set_path: Path,
    evaluation_set_id: str,
    active_model_version: str,
    active_checkpoint_path: Path,
    candidate_version: str,
    candidate_checkpoint_path: Path,
    artifacts: LifecycleArtifacts,
    state: CycleState,
    active_runtime: ActiveRuntimeModel,
    active_decision_thresholds: dict[str, Any],
    gt_masks_manifest: Path,
) -> dict[str, Any]:
    started_at = datetime.now(UTC)
    active = evaluate_progressive_model_on_set(
        args,
        model_version=active_model_version,
        checkpoint_path=active_checkpoint_path,
        evaluation_set_path=evaluation_set_path,
        output_dir=cycle_dir / "evaluation" / panel_name / "active_before",
        evaluation_set_id=evaluation_set_id,
        cache_root=cycle_dir.parents[1] / "prediction_cache",
        gt_masks_manifest=gt_masks_manifest,
    )
    candidate = evaluate_progressive_model_on_set(
        args,
        model_version=candidate_version,
        checkpoint_path=candidate_checkpoint_path,
        evaluation_set_path=evaluation_set_path,
        output_dir=cycle_dir / "evaluation" / panel_name / "candidate",
        evaluation_set_id=evaluation_set_id,
        cache_root=cycle_dir.parents[1] / "prediction_cache",
        gt_masks_manifest=gt_masks_manifest,
    )
    active_calibrated_thresholds = thresholds_from_evaluation_scores(
        active["metrics_path"],
        evaluation_set_id=evaluation_set_id,
        model_version=active_model_version,
        role="active_before",
    )
    active = apply_decision_thresholds_to_evaluation(
        active,
        decision_thresholds=active_calibrated_thresholds,
    )
    candidate_decision_thresholds = thresholds_from_evaluation_scores(
        candidate["metrics_path"],
        evaluation_set_id=evaluation_set_id,
        model_version=candidate_version,
        role="candidate",
    )
    candidate = apply_decision_thresholds_to_evaluation(
        candidate,
        decision_thresholds=candidate_decision_thresholds,
    )
    completed_at = datetime.now(UTC)
    duration = (completed_at - started_at).total_seconds()
    for role, payload in (("active_before", active), ("candidate", candidate)):
        if payload.get("eval_inference_seconds") is not None and not payload.get("cache_hit"):
            record_timing(
                artifacts,
                "eval_inference",
                duration_seconds=float(payload.get("eval_inference_seconds") or 0.0),
                cycle_id=cycle_dir.name,
                panel=panel_name,
                role=role,
                cache_status=payload.get("cache_status"),
            )
        metric_timings = payload.get("metric_timings") or {}
        if metric_timings.get("aupimo_compute_seconds") is not None:
            record_timing(
                artifacts,
                "aupimo_compute",
                duration_seconds=float(metric_timings.get("aupimo_compute_seconds") or 0.0),
                cycle_id=cycle_dir.name,
                panel=panel_name,
                role=role,
                cache_status=payload.get("cache_status"),
            )
        if metric_timings.get("pixel_rank_metrics_seconds") is not None:
            record_timing(
                artifacts,
                "pixel_rank_metrics",
                duration_seconds=float(metric_timings.get("pixel_rank_metrics_seconds") or 0.0),
                cycle_id=cycle_dir.name,
                panel=panel_name,
                role=role,
                cache_status=payload.get("cache_status"),
            )
    record_timing(
        artifacts,
        "comparative_eval",
        duration_seconds=duration,
        cycle_id=cycle_dir.name,
        panel=panel_name,
        active_cache_status=active.get("cache_status"),
        candidate_cache_status=candidate.get("cache_status"),
        active_metric_timings=active.get("metric_timings"),
        candidate_metric_timings=candidate.get("metric_timings"),
    )
    for role, payload in (("active_before", active), ("candidate", candidate)):
        record_lifecycle_event(
            artifacts,
            state,
            active_runtime,
            "prediction_cache_hit" if payload.get("cache_hit") else "prediction_cache_miss",
            cycle_id=cycle_dir.name,
            panel=panel_name,
            role=role,
            model_version=payload.get("model_version"),
            cache_key=payload.get("cache_key"),
            cache_status=payload.get("cache_status"),
        )
    record_lifecycle_event(
        artifacts,
        state,
        active_runtime,
        "evaluation_completed",
        cycle_id=cycle_dir.name,
        panel=panel_name,
        active_model_version=active_model_version,
        candidate_version=candidate_version,
        duration_seconds=duration,
    )
    selected_metric = select_comparable_business_metric(active["metrics"], candidate["metrics"])
    if selected_metric is None:
        return {
            "panel": panel_name,
            "metrics": candidate["metrics"],
            "active_metrics_on_eval_set": active["metrics"],
            "candidate_metrics_on_eval_set": candidate["metrics"],
            "active_metric_value": None,
            "candidate_metric_value": None,
            "metric_delta": None,
            "selected_metric": None,
            "selected_metric_value": None,
            "gate_decision": "rejected",
            "gate_reason": "rejected_missing_comparable_metric",
            "promotion_status": "rejected_missing_comparable_metric",
            "active_eval_metrics_path": active["metrics_path"],
            "candidate_eval_metrics_path": candidate["metrics_path"],
            "active_cache_status": active.get("cache_status"),
            "candidate_cache_status": candidate.get("cache_status"),
            "active_cache_key": active.get("cache_key"),
            "candidate_cache_key": candidate.get("cache_key"),
            "cache_status": candidate.get("cache_status"),
            "cache_key": candidate.get("cache_key"),
            "cache_hit": candidate.get("cache_hit"),
            "cache_source": candidate.get("cache_source"),
            "candidate_decision_thresholds": candidate_decision_thresholds,
            "active_decision_thresholds": active_calibrated_thresholds,
            "prediction_schema_version": candidate.get("prediction_schema_version"),
            "candidate_metric_timings": candidate.get("metric_timings", {}),
            "active_metric_timings": active.get("metric_timings", {}),
            "threshold_source": (
                candidate_decision_thresholds.get("threshold_source") if candidate_decision_thresholds else ""
            ),
            "operational_alerts": ["missing_comparable_business_metric"],
            "evaluation_started_at": started_at.isoformat(),
            "evaluation_completed_at": completed_at.isoformat(),
            "evaluation_duration_seconds": (completed_at - started_at).total_seconds(),
            "defective_count": count_defective_rows(evaluation_set_path),
            "per_class_regressions": [],
        }
    active_value = float(active["metrics"][selected_metric])
    candidate_value = float(candidate["metrics"][selected_metric])
    delta = candidate_value - active_value
    active_false_negatives = int(active["metrics"].get("false_negatives") or 0)
    candidate_false_negatives = int(candidate["metrics"].get("false_negatives") or 0)
    operational_alerts: list[str] = []
    if candidate_false_negatives > active_false_negatives:
        operational_alerts.append("candidate_increases_false_negatives")
    active_good_alert_rate = float(active["metrics"].get("good_alert_rate") or 0.0)
    candidate_good_alert_rate = float(candidate["metrics"].get("good_alert_rate") or 0.0)
    active_good_red_rate = float(active["metrics"].get("good_red_rate") or 0.0)
    candidate_good_red_rate = float(candidate["metrics"].get("good_red_rate") or 0.0)
    if candidate_decision_thresholds is None:
        operational_alerts.append("missing_candidate_runtime_thresholds")
    if active_calibrated_thresholds is None:
        operational_alerts.append("missing_active_runtime_thresholds")
    return {
        "panel": panel_name,
        "metrics": candidate["metrics"],
        "active_metrics_on_eval_set": active["metrics"],
        "candidate_metrics_on_eval_set": candidate["metrics"],
        "active_metric_value": active_value,
        "candidate_metric_value": candidate_value,
        "metric_delta": delta,
        "selected_metric": selected_metric,
        "selected_metric_value": candidate_value,
        "gate_decision": "not_aggregated",
        "gate_reason": "panel_evaluated",
        "promotion_status": "panel_evaluated",
        "active_eval_metrics_path": active["metrics_path"],
        "candidate_eval_metrics_path": candidate["metrics_path"],
        "active_cache_status": active.get("cache_status"),
        "candidate_cache_status": candidate.get("cache_status"),
        "active_cache_key": active.get("cache_key"),
        "candidate_cache_key": candidate.get("cache_key"),
        "cache_status": candidate.get("cache_status"),
        "cache_key": candidate.get("cache_key"),
        "cache_hit": candidate.get("cache_hit"),
        "cache_source": candidate.get("cache_source"),
        "candidate_decision_thresholds": candidate_decision_thresholds,
        "active_decision_thresholds": active_calibrated_thresholds,
        "prediction_schema_version": candidate.get("prediction_schema_version"),
        "candidate_metric_timings": candidate.get("metric_timings", {}),
        "active_metric_timings": active.get("metric_timings", {}),
        "threshold_source": (
            candidate_decision_thresholds.get("threshold_source") if candidate_decision_thresholds else ""
        ),
        "candidate_false_negatives": candidate_false_negatives,
        "active_false_negatives": active_false_negatives,
        "candidate_good_alert_rate": candidate_good_alert_rate,
        "active_good_alert_rate": active_good_alert_rate,
        "candidate_good_red_rate": candidate_good_red_rate,
        "active_good_red_rate": active_good_red_rate,
        "candidate_good_red_count": int(candidate["metrics"].get("good_red_count") or 0),
        "active_good_red_count": int(active["metrics"].get("good_red_count") or 0),
        "active_per_class_metrics": active.get("per_class_metrics", {}),
        "candidate_per_class_metrics": candidate.get("per_class_metrics", {}),
        "candidate_aupimo_stability": candidate.get("aupimo_stability", {}),
        "active_aupimo_stability": active.get("aupimo_stability", {}),
        "candidate_predictions_path": candidate.get("predictions_path"),
        "active_predictions_path": active.get("predictions_path"),
        "evaluation_started_at": started_at.isoformat(),
        "evaluation_completed_at": completed_at.isoformat(),
        "evaluation_duration_seconds": (completed_at - started_at).total_seconds(),
        "operational_alerts": operational_alerts,
        "defective_count": count_defective_rows(evaluation_set_path),
        "per_class_regressions": [],
    }


def evaluate_progressive_promotion_comparison(
    args: argparse.Namespace,
    *,
    cycle_dir: Path,
    reference_evaluation_set_path: Path,
    reference_evaluation_set_id: str,
    progressive_evaluation_set_path: Path,
    progressive_evaluation_set_id: str,
    active_model_version: str,
    active_checkpoint_path: Path,
    candidate_version: str,
    candidate_checkpoint_path: Path,
    artifacts: LifecycleArtifacts,
    state: CycleState,
    active_runtime: ActiveRuntimeModel,
) -> dict[str, Any]:
    started_at = datetime.now(UTC)
    reference = evaluate_model_pair_on_panel(
        args,
        cycle_dir=cycle_dir,
        panel_name="reference",
        evaluation_set_path=reference_evaluation_set_path,
        evaluation_set_id=reference_evaluation_set_id,
        active_model_version=active_model_version,
        active_checkpoint_path=active_checkpoint_path,
        candidate_version=candidate_version,
        candidate_checkpoint_path=candidate_checkpoint_path,
        artifacts=artifacts,
        state=state,
        active_runtime=active_runtime,
        active_decision_thresholds=active_runtime.decision_thresholds,
        gt_masks_manifest=args.reference_gt_masks_manifest,
    )
    progressive_defective_count = count_defective_rows(progressive_evaluation_set_path)
    progressive_has_enough_defects = progressive_defective_count >= int(args.progressive_min_defects_for_decision)
    progressive: dict[str, Any] | None = None
    if progressive_has_enough_defects:
        progressive = evaluate_model_pair_on_panel(
            args,
            cycle_dir=cycle_dir,
            panel_name="progressive",
            evaluation_set_path=progressive_evaluation_set_path,
            evaluation_set_id=progressive_evaluation_set_id,
            active_model_version=active_model_version,
            active_checkpoint_path=active_checkpoint_path,
            candidate_version=candidate_version,
            candidate_checkpoint_path=candidate_checkpoint_path,
            artifacts=artifacts,
            state=state,
            active_runtime=active_runtime,
            active_decision_thresholds=active_runtime.decision_thresholds,
            gt_masks_manifest=args.reference_gt_masks_manifest,
        )
    completed_at = datetime.now(UTC)

    reference_delta = reference.get("metric_delta")
    reference_false_negatives_ok = int(reference.get("candidate_false_negatives") or 0) <= int(
        reference.get("active_false_negatives") or 0
    )
    progressive_false_negatives_ok = True
    if progressive is not None:
        progressive_false_negatives_ok = int(progressive.get("candidate_false_negatives") or 0) <= int(
            progressive.get("active_false_negatives") or 0
        )
    active_metrics = reference.get("active_metrics_on_eval_set") or {}
    candidate_metrics = reference.get("candidate_metrics_on_eval_set") or {}
    active_good_red_count = int(active_metrics.get("good_red_count") or 0)
    candidate_good_red_count = int(candidate_metrics.get("good_red_count") or 0)
    good_red_delta = candidate_good_red_count - active_good_red_count
    good_red_ok = good_red_delta <= int(args.max_good_red_regression)
    metric_ok = (
        reference.get("selected_metric") is not None
        and reference_delta is not None
        and float(reference_delta) > float(args.promotion_min_delta)
    )
    candidate_thresholds = reference.get("candidate_decision_thresholds")
    active_thresholds = reference.get("active_decision_thresholds")
    thresholds_ok = candidate_thresholds is not None and active_thresholds is not None
    passed = bool(metric_ok and reference_false_negatives_ok and progressive_false_negatives_ok and good_red_ok and thresholds_ok)

    simplified_gate = {
        "decision": "passed" if passed else "rejected",
        "selected_metric": reference.get("selected_metric"),
        "active_metric_value": reference.get("active_metric_value"),
        "candidate_metric_value": reference.get("candidate_metric_value"),
        "metric_delta": reference.get("metric_delta"),
        "metric_ok": metric_ok,
        "false_negatives_ok": reference_false_negatives_ok,
        "progressive_false_negatives_ok": progressive_false_negatives_ok,
        "active_false_negatives": reference.get("active_false_negatives"),
        "candidate_false_negatives": reference.get("candidate_false_negatives"),
        "active_good_red_count": active_good_red_count,
        "candidate_good_red_count": candidate_good_red_count,
        "good_red_delta": good_red_delta,
        "max_good_red_regression": int(args.max_good_red_regression),
        "good_red_ok": good_red_ok,
        "thresholds_ok": thresholds_ok,
        "progressive_monitor_only": not progressive_has_enough_defects,
        "progressive_defective_count": progressive_defective_count,
        "progressive_min_defects_for_decision": int(args.progressive_min_defects_for_decision),
    }

    missing_comparable_metric = reference.get("selected_metric") is None
    if missing_comparable_metric:
        gate_reason = "rejected_missing_comparable_metric"
        promotion_status = "rejected_missing_comparable_metric"
        passed = False
    elif passed:
        gate_reason = "candidate_passed_reference_gate"
        promotion_status = "promoted"
    elif not reference_false_negatives_ok or not progressive_false_negatives_ok:
        gate_reason = "candidate_increases_false_negatives"
        promotion_status = "rejected_operational_guardrail"
    elif not good_red_ok:
        gate_reason = "candidate_increases_good_red_count"
        promotion_status = "rejected_operational_guardrail"
    elif not metric_ok:
        gate_reason = "candidate_regressed_on_reference_panel"
        promotion_status = "rejected_reference_regression"
    elif not thresholds_ok:
        gate_reason = "missing_candidate_runtime_thresholds"
        promotion_status = "rejected_missing_runtime_thresholds"
    else:
        gate_reason = "candidate_rejected_by_panel_gate"
        promotion_status = "rejected_panel_gate"

    operational_alerts = list(reference.get("operational_alerts") or [])
    if progressive is not None:
        operational_alerts.extend(progressive.get("operational_alerts") or [])
    if not progressive_has_enough_defects:
        operational_alerts.append("progressive_panel_not_enough_defects_for_decision")

    return {
        "metrics": reference.get("metrics") or {},
        "active_metrics_on_eval_set": reference.get("active_metrics_on_eval_set") or {},
        "candidate_metrics_on_eval_set": reference.get("candidate_metrics_on_eval_set") or {},
        "reference_active_metrics_on_eval_set": reference.get("active_metrics_on_eval_set") or {},
        "reference_candidate_metrics_on_eval_set": reference.get("candidate_metrics_on_eval_set") or {},
        "progressive_active_metrics_on_eval_set": (progressive or {}).get("active_metrics_on_eval_set") or {},
        "progressive_candidate_metrics_on_eval_set": (progressive or {}).get("candidate_metrics_on_eval_set") or {},
        "reference_selected_metric": reference.get("selected_metric"),
        "progressive_selected_metric": (progressive or {}).get("selected_metric"),
        "selected_metric": reference.get("selected_metric"),
        "selected_metric_value": reference.get("candidate_metric_value"),
        "active_metric_value": reference.get("active_metric_value"),
        "candidate_metric_value": reference.get("candidate_metric_value"),
        "metric_delta": reference.get("metric_delta"),
        "fn_delta": int(reference.get("candidate_false_negatives") or 0) - int(reference.get("active_false_negatives") or 0),
        "good_red_delta": good_red_delta,
        "reference_active_metric_value": reference.get("active_metric_value"),
        "reference_candidate_metric_value": reference.get("candidate_metric_value"),
        "reference_metric_delta": reference.get("metric_delta"),
        "progressive_active_metric_value": (progressive or {}).get("active_metric_value"),
        "progressive_candidate_metric_value": (progressive or {}).get("candidate_metric_value"),
        "progressive_metric_delta": (progressive or {}).get("metric_delta"),
        "gate_decision": "passed" if passed else "rejected",
        "gate_reason": gate_reason,
        "promotion_status": promotion_status,
        "promotion_panel_decision": simplified_gate,
        "simplified_gate": simplified_gate,
        "active_decision_thresholds": active_thresholds,
        "active_eval_metrics_path": reference.get("active_eval_metrics_path"),
        "candidate_eval_metrics_path": reference.get("candidate_eval_metrics_path"),
        "reference_active_eval_metrics_path": reference.get("active_eval_metrics_path"),
        "reference_candidate_eval_metrics_path": reference.get("candidate_eval_metrics_path"),
        "progressive_active_eval_metrics_path": (progressive or {}).get("active_eval_metrics_path"),
        "progressive_candidate_eval_metrics_path": (progressive or {}).get("candidate_eval_metrics_path"),
        "active_cache_status": reference.get("active_cache_status"),
        "candidate_cache_status": reference.get("candidate_cache_status"),
        "active_cache_key": reference.get("active_cache_key"),
        "candidate_cache_key": reference.get("candidate_cache_key"),
        "cache_status": reference.get("cache_status"),
        "cache_key": reference.get("cache_key"),
        "cache_hit": reference.get("cache_hit"),
        "cache_source": reference.get("cache_source"),
        "candidate_decision_thresholds": candidate_thresholds,
        "active_good_alert_rate": reference.get("active_good_alert_rate"),
        "candidate_good_alert_rate": reference.get("candidate_good_alert_rate"),
        "active_good_red_rate": reference.get("active_good_red_rate"),
        "candidate_good_red_rate": reference.get("candidate_good_red_rate"),
        "active_good_red_count": active_good_red_count,
        "candidate_good_red_count": candidate_good_red_count,
        "reference_active_good_alert_rate": reference.get("active_good_alert_rate"),
        "reference_candidate_good_alert_rate": reference.get("candidate_good_alert_rate"),
        "reference_active_good_red_rate": reference.get("active_good_red_rate"),
        "reference_candidate_good_red_rate": reference.get("candidate_good_red_rate"),
        "prediction_schema_version": reference.get("prediction_schema_version"),
        "candidate_metric_timings": reference.get("candidate_metric_timings", {}),
        "active_metric_timings": reference.get("active_metric_timings", {}),
        "threshold_source": candidate_thresholds.get("threshold_source") if candidate_thresholds else "",
        "candidate_false_negatives": reference.get("candidate_false_negatives"),
        "active_false_negatives": reference.get("active_false_negatives"),
        "reference_candidate_false_negatives": reference.get("candidate_false_negatives"),
        "reference_active_false_negatives": reference.get("active_false_negatives"),
        "active_per_class_metrics": reference.get("active_per_class_metrics", {}),
        "candidate_per_class_metrics": reference.get("candidate_per_class_metrics", {}),
        "reference_active_per_class_metrics": reference.get("active_per_class_metrics", {}),
        "reference_candidate_per_class_metrics": reference.get("candidate_per_class_metrics", {}),
        "per_class_regressions": reference.get("per_class_regressions") or [],
        "candidate_aupimo_stability": reference.get("candidate_aupimo_stability", {}),
        "active_aupimo_stability": reference.get("active_aupimo_stability", {}),
        "reference_candidate_aupimo_stability": reference.get("candidate_aupimo_stability", {}),
        "progressive_candidate_aupimo_stability": (progressive or {}).get("candidate_aupimo_stability", {}),
        "candidate_predictions_path": reference.get("candidate_predictions_path"),
        "active_predictions_path": reference.get("active_predictions_path"),
        "reference_candidate_predictions_path": reference.get("candidate_predictions_path"),
        "progressive_candidate_predictions_path": (progressive or {}).get("candidate_predictions_path"),
        "evaluation_started_at": started_at.isoformat(),
        "evaluation_completed_at": completed_at.isoformat(),
        "evaluation_duration_seconds": (completed_at - started_at).total_seconds(),
        "operational_alerts": operational_alerts,
    }


def evaluate_progressive_model_on_set(
    args: argparse.Namespace,
    *,
    model_version: str,
    checkpoint_path: Path,
    evaluation_set_path: Path,
    output_dir: Path,
    evaluation_set_id: str,
    cache_root: Path,
    decision_thresholds: dict[str, Any] | None = None,
    gt_masks_manifest: Path | None = None,
) -> dict[str, Any]:
    threshold_orange = float((decision_thresholds or {}).get("threshold_orange") or 0.02)
    threshold_red = float((decision_thresholds or {}).get("threshold_red") or 0.05)
    cache_key = prediction_cache_key(
        model_version=model_version,
        checkpoint_path=checkpoint_path,
        evaluation_set_path=evaluation_set_path,
        evaluation_set_id=evaluation_set_id,
        threshold_orange=threshold_orange,
        threshold_red=threshold_red,
        roi_signature=f"functional_surface_prediction:{CANONICAL_FEATURE_AE_PREPROCESSING.roi_threshold:.12g}",
        calibration_signature="none",
    )
    cached = load_prediction_cache(cache_root, cache_key, output_dir)
    if cached is not None:
        cached.update(
            {
                "model_version": model_version,
                "checkpoint_path": str(checkpoint_path),
                "cache_status": "hit",
                "cache_hit": True,
                "cache_key": cache_key,
                "cache_source": str(cache_root / cache_key),
            }
        )
        return cached
    result = evaluate_feature_ae_checkpoint(
        FeatureAEEvaluationConfig(
            checkpoint_path=checkpoint_path,
            manifest_path=evaluation_set_path,
            image_root=args.image_root,
            output_dir=output_dir,
            gt_masks_manifest=gt_masks_manifest or args.reference_gt_masks_manifest,
            validation_set_id=evaluation_set_path.stem,
            batch_size=args.batch_size,
            device=args.device,
            layer_weights=CANONICAL_FEATURE_AE_PREPROCESSING.layer_weights,
            roi_threshold=CANONICAL_FEATURE_AE_PREPROCESSING.roi_threshold,
            score_smoothing=CANONICAL_FEATURE_AE_PREPROCESSING.score_smoothing,
            score_image=CANONICAL_FEATURE_AE_PREPROCESSING.score_image,
            topk_fraction=CANONICAL_FEATURE_AE_PREPROCESSING.topk_fraction,
            threshold_orange=threshold_orange,
            threshold_red=threshold_red,
        )
    )
    params = json.loads((output_dir / "params.json").read_text(encoding="utf-8")) if (output_dir / "params.json").is_file() else {}
    store_prediction_cache(cache_root, cache_key, output_dir)
    metrics = {
        metric: float(value)
        for metric, value in (result.get("metrics") or {}).items()
        if value is not None
        and metric
        in {
            *FEATURE_AE_BUSINESS_METRIC_PRIORITY,
            "pixel_auroc",
            "image_recall",
            "false_negatives",
            "orange_rate",
        }
    }
    return {
        "model_version": model_version,
        "checkpoint_path": str(checkpoint_path),
        "metrics": metrics,
        "per_class_metrics": result.get("per_class_metrics") or {},
        "aupimo_stability": result.get("aupimo_stability") or {},
        "predictions_path": result.get("predictions_path"),
        "metrics_path": str(output_dir / "metrics.json"),
        "params_path": str(output_dir / "params.json"),
        "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
        "metric_timings": result.get("metric_timings") or {},
        "eval_inference_seconds": params.get("duration_seconds"),
        "cache_status": "miss_stored",
        "cache_hit": False,
        "cache_key": cache_key,
        "cache_source": str(cache_root / cache_key),
    }


def prediction_cache_key(
    *,
    model_version: str,
    checkpoint_path: Path,
    evaluation_set_path: Path,
    evaluation_set_id: str,
    threshold_orange: float,
    threshold_red: float,
    roi_signature: str,
    calibration_signature: str,
) -> str:
    payload = {
        "model_version": model_version,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "evaluation_set_id": evaluation_set_id,
        "evaluation_set_sha256": sha256_file(evaluation_set_path),
        "score_contract_version": CANONICAL_FEATURE_AE_PREPROCESSING.version,
        "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
        "roi_signature": roi_signature,
        "calibration_signature": calibration_signature,
        "threshold_signature": f"{threshold_orange:.12g}:{threshold_red:.12g}",
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_prediction_cache(cache_root: Path, cache_key: str, output_dir: Path) -> dict[str, Any] | None:
    cache_dir = cache_root / cache_key
    metrics_path = cache_dir / "metrics.json"
    predictions_path = cache_dir / "predictions.npz"
    if not metrics_path.is_file() or not predictions_path.is_file():
        return None
    params_path = cache_dir / "params.json"
    if not params_path.is_file():
        return None
    params: dict[str, Any] = json.loads(params_path.read_text(encoding="utf-8"))
    if params.get("prediction_schema_version") != PREDICTION_SCHEMA_VERSION:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("metrics.json", "predictions.npz", "params.json"):
        source = cache_dir / name
        if source.is_file():
            shutil.copy2(source, output_dir / name)
    payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics = {
        metric: float(value)
        for metric, value in (payload.get("metrics") or {}).items()
        if value is not None
        and metric
        in {
            *FEATURE_AE_BUSINESS_METRIC_PRIORITY,
            "pixel_auroc",
            "image_recall",
            "false_negatives",
            "orange_rate",
        }
    }
    return {
        "metrics": metrics,
        "per_class_metrics": payload.get("per_class_metrics") or {},
        "aupimo_stability": payload.get("aupimo_stability") or {},
        "predictions_path": str(output_dir / "predictions.npz"),
        "metrics_path": str(output_dir / "metrics.json"),
        "params_path": str(output_dir / "params.json"),
        "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
        "metric_timings": payload.get("metric_timings") or {},
        "eval_inference_seconds": params.get("duration_seconds"),
    }


def store_prediction_cache(cache_root: Path, cache_key: str, output_dir: Path) -> None:
    cache_dir = cache_root / cache_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name in ("metrics.json", "predictions.npz", "params.json"):
        source = output_dir / name
        if source.is_file():
            shutil.copy2(source, cache_dir / name)
    index_path = cache_root / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {}
    index[cache_key] = {
        "cache_dir": str(cache_dir),
        "created_at": datetime.now(UTC).isoformat(),
    }
    write_json(index_path, index)


def select_comparable_business_metric(
    active_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
) -> str | None:
    for metric in FEATURE_AE_BUSINESS_METRIC_PRIORITY:
        if active_metrics.get(metric) is not None and candidate_metrics.get(metric) is not None:
            return metric
    return None


def apply_decision_thresholds_to_evaluation(
    evaluation: dict[str, Any],
    *,
    decision_thresholds: dict[str, Any] | None,
) -> dict[str, Any]:
    """Refresh operational metrics from per-image scores without re-running inference."""
    if not decision_thresholds:
        return evaluation
    metrics_path = Path(str(evaluation.get("metrics_path") or ""))
    if not metrics_path.is_file():
        return evaluation
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    images = payload.get("images") or payload.get("predictions") or []
    labels = [bool(record.get("is_defective")) for record in images if record.get("score") is not None]
    scores = [float(record["score"]) for record in images if record.get("score") is not None]
    existing_metrics = payload.get("metrics") or {}
    if labels and not any(labels) and int(existing_metrics.get("false_negatives") or 0) > 0:
        return evaluation
    decision = compute_decision_metrics(
        labels,
        scores,
        threshold_orange=float(decision_thresholds["threshold_orange"]),
        threshold_red=float(decision_thresholds["threshold_red"]),
    )
    payload.setdefault("metrics", {})
    payload["metrics"]["image_recall"] = decision["recall"]
    payload["metrics"]["false_negatives"] = decision["false_negatives"]
    payload["metrics"]["false_positive_count"] = decision["false_positive_count"]
    payload["metrics"]["alert_count"] = decision["alert_count"]
    payload["metrics"]["red_count"] = decision["red_count"]
    payload["metrics"]["good_alert_count"] = decision["good_alert_count"]
    payload["metrics"]["good_red_count"] = decision["good_red_count"]
    payload["metrics"]["orange_rate"] = decision["orange_rate"]
    payload["metrics"]["alert_rate"] = decision["alert_rate"]
    payload["metrics"]["red_rate"] = decision["red_rate"]
    payload["metrics"]["good_alert_rate"] = decision["good_alert_rate"]
    payload["metrics"]["good_red_rate"] = decision["good_red_rate"]
    payload["metrics"]["latency_ms"] = decision["latency_ms"]
    payload["decision_thresholds"] = decision_thresholds
    threshold_orange = float(decision_thresholds["threshold_orange"])
    threshold_red = float(decision_thresholds["threshold_red"])
    for record in images:
        if record.get("score") is None:
            continue
        score = float(record["score"])
        is_defective = bool(record.get("is_defective"))
        is_alert = score >= threshold_orange
        is_red = score >= threshold_red
        record["threshold_orange"] = threshold_orange
        record["threshold_red"] = threshold_red
        record["is_alert"] = is_alert
        record["is_red"] = is_red
        record["decision"] = "red" if is_red else ("orange" if is_alert else "green")
        record["is_false_positive"] = bool((not is_defective) and is_alert)
        record["is_false_negative"] = bool(is_defective and not is_alert)
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    evaluation_metrics = dict(evaluation.get("metrics") or {})
    evaluation_metrics.update(
        {
            "image_recall": float(decision["recall"]),
            "false_negatives": float(decision["false_negatives"]),
            "false_positive_count": float(decision["false_positive_count"]),
            "alert_count": float(decision["alert_count"]),
            "red_count": float(decision["red_count"]),
            "good_alert_count": float(decision["good_alert_count"]),
            "good_red_count": float(decision["good_red_count"]),
            "orange_rate": float(decision["orange_rate"]),
            "alert_rate": float(decision["alert_rate"]),
            "red_rate": float(decision["red_rate"]),
            "good_alert_rate": float(decision["good_alert_rate"]),
            "good_red_rate": float(decision["good_red_rate"]),
            "latency_ms": float(decision["latency_ms"]),
        }
    )
    evaluation["metrics"] = evaluation_metrics
    return evaluation


def count_defective_rows(manifest_path: Path) -> int:
    if not manifest_path.is_file():
        return 0
    count = 0
    with manifest_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            oracle = str(row.get("oracle_verdict") or "").lower()
            label = str(row.get("label") or "").lower()
            is_defective = str(row.get("is_defective") or "").lower()
            if oracle in {"defective", "defaut", "defectueux", "non_conforme"} or label == "defective" or is_defective == "true":
                count += 1
    return count


def per_class_regressions(
    active_per_class: dict[str, Any],
    candidate_per_class: dict[str, Any],
    *,
    metric: str | None,
    min_delta: float,
) -> list[dict[str, Any]]:
    if not metric:
        return []
    regressions: list[dict[str, Any]] = []
    for source_class, active_metrics in active_per_class.items():
        candidate_metrics = candidate_per_class.get(source_class)
        if not isinstance(active_metrics, dict) or not isinstance(candidate_metrics, dict):
            continue
        active_value = active_metrics.get(metric)
        candidate_value = candidate_metrics.get(metric)
        if active_value is None or candidate_value is None:
            continue
        delta = float(candidate_value) - float(active_value)
        if delta < -abs(min_delta):
            regressions.append(
                {
                    "source_class": source_class,
                    "metric": metric,
                    "active_value": float(active_value),
                    "candidate_value": float(candidate_value),
                    "delta": delta,
                }
            )
    return regressions


def thresholds_from_evaluation_scores(
    metrics_path: str | Path,
    *,
    evaluation_set_id: str,
    model_version: str = "",
    role: str = "",
    orange_quantile: float = 0.95,
    red_quantile: float = 0.99,
) -> dict[str, Any] | None:
    path = Path(metrics_path)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    images = payload.get("images") or payload.get("predictions") or []
    conforming_scores = [
        float(record["score"])
        for record in images
        if record.get("score") is not None and not bool(record.get("is_defective"))
    ]
    if not conforming_scores:
        return None
    threshold_orange = _interpolated_quantile(conforming_scores, orange_quantile)
    threshold_red = max(threshold_orange, _interpolated_quantile(conforming_scores, red_quantile))
    calibration_signature = (
        f"{CANONICAL_FEATURE_AE_PREPROCESSING.version}:"
        f"{evaluation_set_id}:{model_version}:{role}:"
        f"good_quantiles:{orange_quantile:.6g}:{red_quantile:.6g}"
    )
    return {
        "threshold_orange": threshold_orange,
        "threshold_red": threshold_red,
        "threshold_source": f"panel_good_quantiles:{evaluation_set_id}:{role}",
        "method": "panel_good_quantiles",
        "score_contract_version": CANONICAL_FEATURE_AE_PREPROCESSING.version,
        "orange_quantile": orange_quantile,
        "red_quantile": red_quantile,
        "model_version": model_version,
        "role": role,
        "calibration_signature": calibration_signature,
    }


def _interpolated_quantile(values: list[float], quantile: float) -> float:
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * quantile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = position - lower_index
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    return float(lower_value + (upper_value - lower_value) * fraction)


def register_promoted_cycle(
    args: argparse.Namespace,
    state: CycleState,
    cycle: dict[str, Any],
) -> dict[str, Any]:
    run_id = str(cycle.get("mlflow_run_id") or "")
    if not run_id:
        return {"registry_status": "skipped", "registry_reason": "missing_mlflow_run_id"}
    try:
        result = register_run_to_model(
            run_id=run_id,
            scenario_id=state.scenario_id,
            stage=args.target_stage,
        )
    except Exception as exc:
        if args.require_mlflow_registry:
            raise
        status = "failed_missing_mlflow_model" if "missing_mlflow_model_artifact" in str(exc) else "failed"
        return {
            "registry_status": status,
            "registry_reason": str(exc),
            "registered_model_name": registered_model_name(state.scenario_id),
            "registry_stage": args.target_stage,
            "registry_source_of_truth": "mlflow_registry",
        }
    return {
        "registry_status": "registered",
        "registered_model_name": result.get("registered_model_name"),
        "registered_model_version": result.get("version"),
        "registry_alias": result.get("alias") or result.get("stage") or args.target_stage,
        "registry_stage": result.get("stage") or args.target_stage,
        "registry_source_of_truth": result.get("source_of_truth") or "mlflow_registry",
    }


def tag_mlflow_promotion_evidence(cycle: dict[str, Any]) -> None:
    run_id = str(cycle.get("mlflow_run_id") or "")
    if not run_id:
        return
    try:
        import mlflow

        client = mlflow.tracking.MlflowClient()
        for key in [
            "promotion_policy",
            "active_model_before",
            "candidate_version",
            "evaluation_set_id",
            "evaluation_seen_events",
            "selected_metric",
            "active_metric_value",
            "candidate_metric_value",
            "metric_delta",
            "gate_decision",
            "promotion_status",
            "registered_model_name",
            "registry_stage",
            "registry_status",
            "activated_for_next_events",
            "activation_scope",
            "threshold_source",
            "active_false_negatives",
            "candidate_false_negatives",
        ]:
            value = cycle.get(key)
            if value is not None:
                client.set_tag(run_id, key, str(value))
    except Exception:
        return


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


def _best_candidate_seen(cycles: list[dict[str, Any]]) -> str | None:
    candidates = [
        cycle
        for cycle in cycles
        if cycle.get("candidate_metric_value") is not None and cycle.get("selected_metric") in FEATURE_AE_BUSINESS_METRIC_PRIORITY
    ]
    if not candidates:
        return None
    best = max(candidates, key=lambda cycle: float(cycle["candidate_metric_value"]))
    return str(best.get("candidate_version") or "")


def resolve_candidate_initial_checkpoint(args: argparse.Namespace, *, active_runtime: ActiveRuntimeModel) -> Path | None:
    explicit = getattr(args, "candidate_init_checkpoint", None)
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"--candidate-init-checkpoint path does not exist: {path}")
        return path
    if args.candidate_init_policy == "fresh":
        return None
    if args.candidate_init_policy == "active":
        return active_runtime.checkpoint
    return resolve_feature_ae_checkpoint(DEFAULT_FEATURE_AE_MODEL_VERSION, strict_checksum=True)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return [dict(row) for row in csv.DictReader(file)]


def row_relative_path(row: dict[str, str]) -> str:
    return str(row.get("relative_path") or row.get("relative_paths") or "")


def row_source_class(row: dict[str, str]) -> str:
    return str(row.get("source_class") or row.get("source_classes") or "unknown")


def is_good_training_row(row: dict[str, str]) -> bool:
    label = str(row.get("label") or "").lower()
    oracle = str(row.get("oracle_verdict") or "").lower()
    is_defective = str(row.get("is_defective") or "").lower()
    return label == "good" or oracle == "conforme" or is_defective == "false"


def cap_rows_by_source_class(rows: list[dict[str, str]], max_per_class: int) -> list[dict[str, str]]:
    if max_per_class <= 0:
        return []
    counts: dict[str, int] = {}
    capped: list[dict[str, str]] = []
    for row in rows:
        source_class = row_source_class(row)
        if counts.get(source_class, 0) >= max_per_class:
            continue
        counts[source_class] = counts.get(source_class, 0) + 1
        capped.append(row)
    return capped


def normalize_training_row(row: dict[str, str], *, dataset_snapshot_id: str, scenario_id: str, source: str) -> dict[str, str]:
    relative_path = row_relative_path(row)
    image_id = str(row.get("image_id") or row.get("image_ids") or Path(relative_path).stem)
    normalized = dict(row)
    normalized.update(
        {
            "image_id": image_id,
            "image_ids": image_id,
            "relative_path": relative_path,
            "relative_paths": relative_path,
            "source_class": row_source_class(row),
            "split_set": str(row.get("split_set") or scenario_id),
            "label": "good",
            "is_defective": "false",
            "scenario_id": scenario_id,
            "dataset_version": dataset_snapshot_id,
            "manifest_version": f"{dataset_snapshot_id}_manifest_v001",
            "gt_mask_path": "",
            "oracle_verdict": "conforme",
            "train_eligible": "true",
            "train_eligibility_source": source,
            "quarantine_reason": "",
        }
    )
    return normalized


def write_progressive_training_manifest(
    *,
    seen_snapshot_path: Path,
    anchor_good_manifest: Path,
    output_path: Path,
    dataset_snapshot_id: str,
    scenario_id: str,
    anchor_good_max_per_class: int,
) -> tuple[Path, dict[str, Any]]:
    seen_rows = [
        normalize_training_row(row, dataset_snapshot_id=dataset_snapshot_id, scenario_id=scenario_id, source="oracle_gt_seen_lots")
        for row in read_csv_rows(seen_snapshot_path)
        if is_good_training_row(row)
    ]
    anchor_rows = cap_rows_by_source_class(
        [
            normalize_training_row(row, dataset_snapshot_id=dataset_snapshot_id, scenario_id=scenario_id, source="anchor_good_reference")
            for row in read_csv_rows(anchor_good_manifest)
            if is_good_training_row(row)
        ],
        anchor_good_max_per_class,
    )
    merged: list[dict[str, str]] = []
    seen_relative_paths: set[str] = set()
    for row in [*seen_rows, *anchor_rows]:
        relative_path = row_relative_path(row)
        if not relative_path or relative_path in seen_relative_paths:
            continue
        seen_relative_paths.add(relative_path)
        merged.append(row)
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
    for row in merged:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in merged:
            writer.writerow(row)
    return output_path, {
        "seen_conforming_count": len(seen_rows),
        "anchor_good_count": len(anchor_rows),
        "total_count": len(merged),
        "anchor_good_manifest": str(anchor_good_manifest),
    }


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


def write_seen_evaluation_set(
    events: list[CycleEvent],
    output_path: Path,
    *,
    evaluation_set_id: str,
    scenario_id: str,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "image_ids",
        "relative_path",
        "relative_paths",
        "event_id",
        "piece_event_id",
        "lot_id",
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
        "roi_mask_path",
        "roi_probability_path",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            image_id = Path(event.relative_path).stem
            is_defective = event.oracle_verdict != "conforme"
            writer.writerow(
                {
                    "image_id": image_id,
                    "image_ids": image_id,
                    "relative_path": event.relative_path,
                    "relative_paths": event.relative_path,
                    "event_id": event.event_id,
                    "piece_event_id": event.piece_event_id,
                    "lot_id": event.lot_id,
                    "source_class": event.source_class,
                    "split_set": "progressive_eval",
                    "label": "defective" if is_defective else "good",
                    "is_defective": str(is_defective).lower(),
                    "scenario_id": scenario_id,
                    "dataset_version": event.dataset_version,
                    "manifest_version": evaluation_set_id,
                    "gt_mask_path": event.gt_mask_path,
                    "oracle_verdict": event.oracle_verdict,
                    "train_eligible": "false",
                    "train_eligibility_source": "progressive_eval_same_sample",
                    "quarantine_reason": "" if not is_defective else "oracle_gt_defect",
                    "roi_mask_path": event.roi_mask_path,
                    "roi_probability_path": event.roi_probability_path,
                }
            )
    return output_path



def train_progressive_candidate(
    args: argparse.Namespace,
    candidate_version: str,
    manifest_path: Path,
    dataset_snapshot_id: str,
    *,
    initial_checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    run_dir = Path(".cache/iqa/models") / candidate_version
    _reset_generated_progressive_candidate_run_dir(run_dir)
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
        initial_checkpoint_path=initial_checkpoint_path,
        initial_checkpoint_policy=args.candidate_init_policy,
        run_name=f"{candidate_version}_{args.target_stage}",
        device=args.device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_steps=args.max_steps,
        metric_eval_manifest_path=VALIDATION_MANIFEST,
        gt_masks_manifest=VALIDATION_GT_MASKS_MANIFEST,
        metric_eval_device=args.device,
        metric_eval_every_epochs=1,
        metric_eval_start_epoch=1,
        metric_eval_calibrate_normal=False,
        metric_eval_layer_weights={"layer2": 0.65, "layer3": 0.35},
        metric_eval_apply_score_region_to_map=True,
        require_business_metric_for_early_stopping=True,
    )
    return train_feature_ae_with_mlflow_logging(config, git_commit=_git_commit())


def _reset_generated_progressive_candidate_run_dir(run_dir: Path) -> None:
    """Clear stale artifacts for deterministic progressive candidate versions."""
    if not run_dir.exists():
        return
    if run_dir.resolve().parent != (Path(".cache/iqa/models")).resolve():
        raise ValueError(f"refusing to reset unexpected candidate run dir: {run_dir}")
    shutil.rmtree(run_dir)


def resolve_trigger_warmstart_checkpoint(args: argparse.Namespace) -> Path | None:
    """Pick the warm-start checkpoint for a drift-triggered retrain.

    Order: explicit ``--candidate-init-checkpoint`` wins; otherwise the demo
    warm-start resolver maps ``--triggering-class`` to a pre-baked checkpoint
    (``configs/demo_warmstart_checkpoints.yaml``). ``None`` → train from scratch.
    """
    explicit = getattr(args, "candidate_init_checkpoint", None)
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"--candidate-init-checkpoint path does not exist: {path}")
        return path
    triggering_class = getattr(args, "triggering_class", None)
    if triggering_class:
        resolved = resolve_warmstart_checkpoint(triggering_class)
        if resolved:
            path = Path(resolved)
            if not path.is_file():
                raise FileNotFoundError(
                    f"warm-start checkpoint for {triggering_class} not found: {path} "
                    "(run scripts.bake_warmstart_checkpoints first)"
                )
            return path
    return None


def train_candidate_on_trigger(args: argparse.Namespace, decision: LifecycleDecision) -> dict[str, Any]:
    candidate_version = decision.candidate_dataset_version
    if not candidate_version:
        raise ValueError("lifecycle decision did not provide a candidate_dataset_version")
    manifest_path = Path("data/model_datasets") / f"{candidate_version}.csv"
    run_dir = Path(".cache/iqa/models") / candidate_version
    run_dir.mkdir(parents=True, exist_ok=True)
    init_checkpoint = resolve_trigger_warmstart_checkpoint(args)
    print(f"train-on-trigger warm-start init: {init_checkpoint or 'fresh (no warm-start)'}")
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
        # Distinguishable per-retrain name: the warm-start candidate_version is a
        # constant (feature_ae_good_v003), so name by the triggering class + a
        # timestamp instead -- otherwise every drift retrain shows the same name in
        # the iqa-model-quality experiment and class2 vs class3 are indistinguishable.
        run_name=f"feature_ae_{getattr(args, 'triggering_class', None) or 'drift'}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
        device=args.device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_steps=args.max_steps,
        initial_checkpoint_path=init_checkpoint,
        initial_checkpoint_policy="fresh" if init_checkpoint is None else "explicit",
        metric_eval_manifest_path=VALIDATION_MANIFEST,
        gt_masks_manifest=VALIDATION_GT_MASKS_MANIFEST,
        metric_eval_device=args.device,
        metric_eval_every_epochs=1,
        metric_eval_start_epoch=1,
        metric_eval_calibrate_normal=False,
        metric_eval_layer_weights={"layer2": 0.65, "layer3": 0.35},
        metric_eval_apply_score_region_to_map=True,
        require_business_metric_for_early_stopping=True,
    )
    result = train_feature_ae_with_mlflow_logging(config, git_commit=_git_commit())
    result["run_dir"] = str(run_dir)
    result["warm_started"] = init_checkpoint is not None
    return result


# Minimum candidate image-AP to promote a warm-started drift retrain. The pre-baked
# warm-start checkpoint scores ~0.87, so this is a floor that rejects a collapsed
# candidate, not a fair active-vs-candidate gate (that lives in progressive-train).
TRIGGER_PROMOTION_MIN_IMAGE_AP = 0.40


def _candidate_business_metrics(run_dir: Path) -> dict[str, float]:
    """Read the candidate's best metric-eval values (image_ap, auroc, aupimo)."""
    best_path = run_dir / "metric_eval_best.json"
    if not best_path.is_file():
        return {}
    try:
        payload = json.loads(best_path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, OSError):
        return {}
    metrics: dict[str, float] = {}
    for key, entry in payload.items():
        if isinstance(entry, dict) and isinstance(entry.get("value"), (int, float)):
            metrics[key] = float(entry["value"])
    return metrics


def _reload_inference_model() -> dict[str, Any]:
    """POST /reload-model so the inference drops its cached AE + PatchCore bank.

    Reaches the inference over the demo network (``IQA_INFERENCE_URL``); the
    rebuilt bank with the new coverage then loads on the next prediction, so the
    newly-covered class recovers to in-domain without restarting the container.
    """
    import urllib.error
    import urllib.request

    base = os.environ.get("IQA_INFERENCE_URL", "http://iqa-inference:8100").rstrip("/")
    req = urllib.request.Request(f"{base}/reload-model", data=b"", method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - internal host
            return {"ok": True, "response": json.loads(resp.read().decode("utf-8"))}
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return {"ok": False, "reason": str(exc)}


def _tag_trigger_run_promotion(
    run_id: str,
    *,
    promotion_status: str,
    triggering_class: str | None,
    image_ap: float | None,
    warm_started: bool,
) -> None:
    """Tag the train-on-trigger MLflow run with the promotion outcome.

    Makes the ``iqa-model-quality`` experiment readable at a glance for an
    audience: ``triggering_class`` (which class caused the retrain),
    ``promotion_status`` (promoted vs rejected_*), and ``is_champion`` (the single
    current serving model -- prior champions are demoted so exactly one is true).
    Add these as columns in the MLflow runs table.
    """
    if not run_id:
        return
    try:
        import mlflow

        client = mlflow.tracking.MlflowClient()
        is_champion = promotion_status == "promoted"
        if is_champion:
            run = client.get_run(run_id)
            for prev in client.search_runs(
                [run.info.experiment_id], filter_string="tags.is_champion = 'true'"
            ):
                if prev.info.run_id != run_id:
                    client.set_tag(prev.info.run_id, "is_champion", "false")
        client.set_tag(run_id, "promotion_status", promotion_status)
        client.set_tag(run_id, "is_champion", "true" if is_champion else "false")
        client.set_tag(run_id, "triggering_class", triggering_class or "unknown")
        client.set_tag(run_id, "warm_started", "true" if warm_started else "false")
        if image_ap is not None:
            client.set_tag(run_id, "gate_image_ap", f"{image_ap:.4f}")
    except Exception as exc:  # noqa: BLE001 - tagging is best-effort observability
        print(f"mlflow trigger tagging skipped: {exc}")


def promote_refresh_reload_on_trigger(
    args: argparse.Namespace,
    state: CycleState,
    train_result: dict[str, Any],
) -> None:
    """Gate, then promote → refresh active artifacts → reload inference.

    Closes the autonomous drift-recovery loop for ``train-on-trigger``: a
    warm-started candidate that clears the floor is promoted, the active
    Feature-AE + PatchCore bank are rebuilt for the extended coverage, and the
    inference is told to reload so the newly-covered class recovers to Vert. The
    final ``promotion_status`` is always tagged back onto the MLflow run.
    """
    run_id = str(train_result.get("run_id") or "")
    run_dir = Path(str(train_result.get("run_dir") or ""))
    metrics = _candidate_business_metrics(run_dir) if run_dir.parts else {}
    image_ap = metrics.get("image_ap")
    warm_started = bool(train_result.get("warm_started"))

    try:
        gate_ok = warm_started and (image_ap is None or image_ap >= TRIGGER_PROMOTION_MIN_IMAGE_AP)
        if not gate_ok:
            state.promotion_status = "rejected_below_floor" if warm_started else "rejected_not_warm_started"
            print(
                f"train-on-trigger NOT promoted ({state.promotion_status}): "
                f"warm_started={warm_started} image_ap={image_ap}"
            )
            return

        state.promotion_status = "promoted"
        state.status = "promoted"
        print(f"train-on-trigger PROMOTED: image_ap={image_ap} triggering_class={state.triggering_class}")

        summary_for_refresh = {
            "promotion_status": "promoted",
            "candidate_checkpoint": state.candidate_checkpoint,
            "triggering_class": state.triggering_class,
            "trigger_reason": state.trigger_decision.trigger_reason if state.trigger_decision else "drift",
        }
        refresh = refresh_active_artifacts(summary_for_refresh, Path(".cache/iqa/models"))
        print(f"refresh_active_artifacts: status={refresh.status} covered={refresh.covered_classes} reason={refresh.reason}")
        if refresh.status != "refreshed":
            state.promotion_status = f"promoted_refresh_{refresh.status}"
            return

        reload_result = _reload_inference_model()
        print(f"inference reload: {reload_result}")
        if not reload_result.get("ok"):
            state.promotion_status = "promoted_reload_failed"
    finally:
        _tag_trigger_run_promotion(
            run_id,
            promotion_status=state.promotion_status,
            triggering_class=state.triggering_class,
            image_ap=image_ap,
            warm_started=warm_started,
        )


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
