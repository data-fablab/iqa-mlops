"""Run a replay-driven IQA lifecycle simulation from real manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
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
    feature_ae_reference_contract_from_payload,
    load_feature_ae_decision_thresholds,
    load_feature_ae_reference_contract,
    load_feature_ae_runtime_contract,
    resolve_feature_ae_checkpoint,
    resolve_roi_segmenter_checkpoint,
)
from iqa.monitoring import LifecycleDecision, LifecycleSignal, evaluate_lifecycle_signal
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
from iqa.training.feature_ae_evaluation import FeatureAEEvaluationConfig, compute_decision_metrics, evaluate_feature_ae_checkpoint
from iqa.training.feature_ae_contracts import CANONICAL_FEATURE_AE_PREPROCESSING, FEATURE_AE_BUSINESS_METRIC_PRIORITY
from iqa.training.mlflow_logging import train_feature_ae_with_mlflow_logging
from iqa.registry import register_run_to_model, registered_model_name

NATURAL_SCENARIO_ID = "production_replay_natural"
NATURAL_TRAIN_SCENARIO_ID = "production_replay_natural_train_v004"
PIECE_B_MINIMAL_SCENARIO_ID = "production_replay_natural_piece_b_minimal"
PIECE_B_FULL_SCENARIO_ID = "production_replay_natural_piece_b_full"
PIECE_B_TO_PIECE_A_P4_DRIFT_SCENARIO_ID = "production_replay_natural_piece_b_to_piece_a_p4_drift"
DRIFT_SCENARIO_ID = "drift_domain_extension"
PIECE_B_MINIMAL_BOOTSTRAP_MANIFEST = Path("data/metadata/feature_ae_bootstrap_piece_b_minimal_v001.csv")
PIECE_B_MINIMAL_VALIDATION_MANIFEST = Path("data/validation/validation_set_piece_b_minimal_v001.csv")
PIECE_B_FULL_VALIDATION_MANIFEST = Path("data/validation/validation_set_piece_b_full_v001.csv")
PIECE_B_FULL_GT_MASKS_MANIFEST = Path("data/validation/validation_gt_masks_piece_b_full_v001.csv")
PIECE_A_P4_CLASSIFICATION_SELECTION_MANIFEST = Path(
    "data/validation/classification_selection_piece_b_to_piece_a_p4_drift_v001.csv"
)
PIECE_A_P4_DEMO_PROMOTION_OBJECTIVE = "piece_a_p4_correction_demo"
PIECE_B_NON_REGRESSION_REPORT_ONLY = "report_only"
PIECE_B_NON_REGRESSION_BLOCKING = "blocking"
REPLAY_PLANS = {
    NATURAL_SCENARIO_ID: Path("data/metadata/casting_flux_replay_plan_natural_v003.csv"),
    NATURAL_TRAIN_SCENARIO_ID: Path("data/metadata/casting_flux_replay_plan_natural_train_v004.csv"),
    PIECE_B_MINIMAL_SCENARIO_ID: Path("data/metadata/casting_flux_replay_plan_piece_b_minimal_v001.csv"),
    PIECE_B_FULL_SCENARIO_ID: Path("data/metadata/casting_flux_replay_plan_piece_b_full_v001.csv"),
    PIECE_B_TO_PIECE_A_P4_DRIFT_SCENARIO_ID: Path(
        "data/metadata/casting_flux_replay_plan_piece_b_to_piece_a_p4_drift_v001.csv"
    ),
    DRIFT_SCENARIO_ID: Path("data/metadata/casting_flux_replay_plan_drift.csv"),
}
CANDIDATE_DATASETS = {
    NATURAL_SCENARIO_ID: "feature_ae_good_mvp_v001",
    NATURAL_TRAIN_SCENARIO_ID: "feature_ae_good_mvp_v001",
    PIECE_B_MINIMAL_SCENARIO_ID: "feature_ae_piece_b_minimal_bootstrap_v001",
    PIECE_B_FULL_SCENARIO_ID: "feature_ae_piece_b_minimal_bootstrap_v001",
    PIECE_B_TO_PIECE_A_P4_DRIFT_SCENARIO_ID: "feature_ae_piece_a_p4_drift_v001",
    DRIFT_SCENARIO_ID: "feature_ae_good_mvp_v001",
}
VALIDATION_MANIFEST = Path("data/validation/validation_set_replay_gate_v002.csv")
VALIDATION_GT_MASKS_MANIFEST = Path("data/validation/validation_gt_masks_v001.csv")
DEFAULT_ANCHOR_GOOD_MANIFEST = Path("data/model_datasets/feature_ae_good_mvp_v001.csv")
DEFAULT_OUTPUT_ROOT = Path(".cache/iqa/replay_lifecycle")
DEFAULT_METRIC_CACHE_ROOT = Path(".cache/iqa/metric_eval_cache")
DEFAULT_MODEL_CACHE_ROOT = Path(".cache/iqa/models")
MIN_VALID_CHECKPOINT_BYTES = 1024
Mode = Literal["decision-only", "train-on-trigger", "progressive-decision", "progressive-train"]
PROGRESSIVE_MODES = {"progressive-decision", "progressive-train"}
ACTIVE_REPLAY_SCENARIOS = Path("data/metadata/replay_scenarios.csv")
PROGRESSIVE_PROMOTION_POLICY = "candidate_must_improve_representative_validation_without_operational_regression"
DUAL_PROMOTION_POLICY = "independent_localization_and_classification_promotion"
LOCALIZATION_MODEL_NAME_BASE = "feature_ae_localization"
CLASSIFICATION_MODEL_NAME_BASE = "feature_ae_classifier"
STABLE_PIECE_B_CLASSIFICATION_MODEL_NAME = registered_model_name(
    PIECE_B_FULL_SCENARIO_ID,
    base_name=CLASSIFICATION_MODEL_NAME_BASE,
)
STABLE_PIECE_B_LOCALIZATION_MODEL_NAME = registered_model_name(
    PIECE_B_FULL_SCENARIO_ID,
    base_name=LOCALIZATION_MODEL_NAME_BASE,
)
LOCALIZATION_METRIC_PRIORITY = ("pixel_aupimo_1e-5_1e-3", "pixel_ap", "pixel_auroc")
CLASSIFICATION_METRIC_PRIORITY = ("image_ap", "image_auroc")
GATE_METRICS_TO_KEEP = {
    *FEATURE_AE_BUSINESS_METRIC_PRIORITY,
    "pixel_auroc",
    "image_auroc",
    "image_ap",
    "image_recall",
    "false_negatives",
    "false_positive_count",
    "good_alert_count",
    "good_red_count",
    "alert_count",
    "red_count",
    "orange_rate",
    "alert_rate",
    "red_rate",
    "good_alert_rate",
    "good_red_rate",
    "latency_ms",
}


@dataclass
class CycleEvent:
    event_id: str
    piece_event_id: str
    lot_id: str
    scenario_id: str
    scenario_phase: str
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
    scenario_phases: set[str] = field(default_factory=set)
    dataset_versions: set[str] = field(default_factory=set)
    event_count: int = 0
    conforming_validated_count: int = 0
    defective_count: int = 0
    roi_fail_count: int = 0
    decisions: dict[str, int] = field(default_factory=dict)

    def add(self, event: CycleEvent) -> None:
        self.event_count += 1
        self.source_classes.add(event.source_class)
        self.scenario_phases.add(event.scenario_phase)
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
            "scenario_phases": sorted(self.scenario_phases),
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
    active_localization_model_initial: str = DEFAULT_FEATURE_AE_MODEL_VERSION
    active_localization_model_final: str = DEFAULT_FEATURE_AE_MODEL_VERSION
    cycles_requested: int = 0
    promotion_min_delta: float = 0.0
    cycles: list[dict[str, Any]] = field(default_factory=list)
    promotion_chain: list[str] = field(default_factory=lambda: [DEFAULT_FEATURE_AE_MODEL_VERSION])
    localization_promotion_chain: list[str] = field(default_factory=lambda: [DEFAULT_FEATURE_AE_MODEL_VERSION])
    classification_promotion_chain: list[str] = field(default_factory=lambda: [DEFAULT_FEATURE_AE_MODEL_VERSION])
    seen_events: list[CycleEvent] = field(default_factory=list)
    active_localization_runtime: ActiveRuntimeModel | None = None

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
                    "active_localization_model_initial": self.active_localization_model_initial,
                    "active_localization_model_final": self.active_localization_model_final,
                    "cycles_requested": self.cycles_requested,
                    "cycles_completed": len(self.cycles),
                    "models_promoted": [
                        str(cycle["candidate_version"])
                        for cycle in self.cycles
                        if cycle.get("promotion_status") in {"promoted", "partially_promoted"}
                    ],
                    "promotion_chain": self.promotion_chain,
                    "localization_promotion_chain": self.localization_promotion_chain,
                    "classification_promotion_chain": self.classification_promotion_chain,
                    "localization_models_promoted": [
                        str(cycle["candidate_version"])
                        for cycle in self.cycles
                        if cycle.get("localization_promotion_status") == "promoted"
                    ],
                    "classification_models_promoted": [
                        str(cycle["candidate_version"])
                        for cycle in self.cycles
                        if cycle.get("classification_promotion_status") == "promoted"
                    ],
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
                    "best_localization_cycle": _best_role_cycle_id(self.cycles, role="localization"),
                    "best_classification_cycle": _best_role_cycle_id(self.cycles, role="classification"),
                    "last_mlflow_run_id": self.mlflow_run_id,
                    "rejected_candidates": [
                        str(cycle["candidate_version"])
                        for cycle in self.cycles
                        if str(cycle.get("promotion_status", "")).startswith("rejected")
                        and cycle.get("localization_promotion_status") != "promoted"
                        and cycle.get("classification_promotion_status") != "promoted"
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
    reference_contract: Any | None = None
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
            "runtime_contract_status": "loaded" if self.reference_contract is not None else "default",
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
    parser.add_argument("--scenario-id", choices=sorted(REPLAY_PLANS), default=NATURAL_TRAIN_SCENARIO_ID)
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
    parser.add_argument("--model-cache-root", type=Path, default=DEFAULT_MODEL_CACHE_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--gate-eval-profile", choices=["fast", "full"], default="fast")
    parser.add_argument("--lifecycle-interval", type=int, default=50)
    parser.add_argument("--max-cycles", type=int)
    parser.add_argument("--target-stage", default="test")
    parser.add_argument("--promotion-min-delta", type=float, default=0.0)
    parser.add_argument("--dual-promotion", action="store_true")
    parser.add_argument("--localization-promotion-min-delta", type=float, default=0.0)
    parser.add_argument("--classification-require-fn-improvement", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--classification-min-image-recall-delta", type=float, default=0.0)
    parser.add_argument("--classification-min-image-ap-delta", type=float, default=0.0)
    parser.add_argument("--require-mlflow-registry", action="store_true")
    parser.add_argument("--anchor-good-manifest", type=Path, default=DEFAULT_ANCHOR_GOOD_MANIFEST)
    parser.add_argument("--anchor-good-max-per-class", type=int, default=256)
    parser.add_argument("--reference-eval-manifest", type=Path, default=VALIDATION_MANIFEST)
    parser.add_argument("--classification-selection-manifest", type=Path)
    parser.add_argument("--reference-gt-masks-manifest", type=Path, default=VALIDATION_GT_MASKS_MANIFEST)
    parser.add_argument("--max-good-red-regression", type=int, default=1)
    parser.add_argument("--candidate-init-policy", choices=["stable_base", "active", "fresh"], default="stable_base")
    parser.add_argument("--initial-classification-registered-model", default="")
    parser.add_argument("--initial-localization-registered-model", default="")
    parser.add_argument(
        "--external-drift-confirmed",
        action="store_true",
        help=(
            "Allow a drift-correction lifecycle for scenarios whose drift was "
            "confirmed by an upstream observation/monitoring DAG."
        ),
    )
    args = parser.parse_args()
    if args.scenario_id == PIECE_B_MINIMAL_SCENARIO_ID:
        if args.anchor_good_manifest == DEFAULT_ANCHOR_GOOD_MANIFEST:
            args.anchor_good_manifest = PIECE_B_MINIMAL_BOOTSTRAP_MANIFEST
        if args.reference_eval_manifest == VALIDATION_MANIFEST:
            args.reference_eval_manifest = PIECE_B_MINIMAL_VALIDATION_MANIFEST
    if args.scenario_id in {PIECE_B_FULL_SCENARIO_ID, PIECE_B_TO_PIECE_A_P4_DRIFT_SCENARIO_ID}:
        if args.anchor_good_manifest == DEFAULT_ANCHOR_GOOD_MANIFEST:
            args.anchor_good_manifest = PIECE_B_MINIMAL_BOOTSTRAP_MANIFEST
        if args.reference_eval_manifest == VALIDATION_MANIFEST:
            args.reference_eval_manifest = PIECE_B_FULL_VALIDATION_MANIFEST
        if args.reference_gt_masks_manifest == VALIDATION_GT_MASKS_MANIFEST:
            args.reference_gt_masks_manifest = PIECE_B_FULL_GT_MASKS_MANIFEST
    if args.scenario_id == PIECE_B_TO_PIECE_A_P4_DRIFT_SCENARIO_ID and args.candidate_init_policy == "active":
        if not args.initial_classification_registered_model:
            args.initial_classification_registered_model = STABLE_PIECE_B_CLASSIFICATION_MODEL_NAME
        if not args.initial_localization_registered_model:
            args.initial_localization_registered_model = STABLE_PIECE_B_LOCALIZATION_MODEL_NAME
        if args.classification_selection_manifest is None:
            args.classification_selection_manifest = PIECE_A_P4_CLASSIFICATION_SELECTION_MANIFEST
    return args


def main() -> None:
    args = parse_args()
    try:
        if args.no_gpu_lock:
            result = run_cycle(args)
        else:
            with gpu_lock(owner="iqa-replay-lifecycle", blocking=args.wait_for_gpu):
                result = run_cycle(args)
    except Exception:
        if getattr(args, "lifecycle_run_id", ""):
            emit_lifecycle_api_event(args, "run_failed")
        raise
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
            "active_classification_runtime_model": active_runtime.to_dict(),
            "active_localization_runtime_model": (
                state.active_localization_runtime.to_dict()
                if state.active_localization_runtime is not None
                else active_runtime.to_dict()
            ),
            "active_model_version": active_runtime.version,
            "active_classification_model_version": active_runtime.version,
            "active_localization_model_version": (
                state.active_localization_runtime.version
                if state.active_localization_runtime is not None
                else active_runtime.version
            ),
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
            "active_classification_runtime_final": active_runtime.to_dict(),
            "active_localization_runtime_final": (
                state.active_localization_runtime.to_dict()
                if state.active_localization_runtime is not None
                else active_runtime.to_dict()
            ),
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


def emit_lifecycle_api_event(
    args: argparse.Namespace,
    event_type: str,
    *,
    metrics: dict[str, Any] | None = None,
    **payload: Any,
) -> None:
    api_url = os.getenv("IQA_API_URL", "").strip().rstrip("/")
    if not api_url:
        return
    body = _lifecycle_api_payload(args, event_type, metrics=metrics, **payload)
    data = json.dumps(body, sort_keys=True).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    token = os.getenv("IQA_SERVICE_TOKEN", "").strip()
    if token:
        headers["X-IQA-Service-Token"] = token
    request = urllib.request.Request(
        f"{api_url}/internal/lifecycle/events",
        data=data,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            response.read()
    except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
        print(
            f"warning: lifecycle API push failed for {event_type}: {type(exc).__name__}: {exc}",
            flush=True,
        )


def _lifecycle_api_payload(
    args: argparse.Namespace,
    event_type: str,
    *,
    metrics: dict[str, Any] | None = None,
    **payload: Any,
) -> dict[str, Any]:
    allowed = {
        "event_type",
        "scenario_id",
        "lifecycle_run_id",
        "cycle_id",
        "epoch",
        "candidate_version",
        "candidate_init_policy",
        "candidate_initial_model_version",
        "active_classification_model_version",
        "active_localization_model_version",
        "candidate_initial_checkpoint_sha256",
        "localization_checkpoint_sha256",
        "classification_checkpoint_sha256",
        "localization_promotion_status",
        "classification_promotion_status",
        "localization_gate_reason",
        "classification_gate_reason",
    }
    body = {
        "event_type": event_type,
        "scenario_id": str(getattr(args, "scenario_id", "") or ""),
        "lifecycle_run_id": str(getattr(args, "lifecycle_run_id", "") or ""),
        "candidate_init_policy": str(getattr(args, "candidate_init_policy", "") or ""),
    }
    body.update({key: value for key, value in payload.items() if key in allowed and value not in (None, "")})
    sanitized_metrics = {
        key: value
        for key, value in (metrics or {}).items()
        if key
        in {
            "pixel_aupimo_1e-5_1e-3",
            "pixel_ap",
            "image_ap",
            "localization_metric_delta",
            "classification_metric_delta",
            "classification_fn_delta",
            "gate_metric_delta",
            "gate_fn_delta",
        }
        and _finite_float(value) is not None
    }
    if sanitized_metrics:
        body["metrics"] = sanitized_metrics
    return body


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


def resolve_registered_initial_runtime(
    args: argparse.Namespace,
    *,
    model_name: str,
    role: Literal["classification", "localization"],
    fallback_thresholds: dict[str, Any],
) -> ActiveRuntimeModel | None:
    if not model_name:
        return None
    artifact_path = "model_classification" if role == "classification" else "model_localization"
    checkpoint_names = (
        ("checkpoint_best_image.pt", "checkpoint_best_image_ap.pt", "checkpoint.pt")
        if role == "classification"
        else (
            "checkpoint_best_localization.pt",
            "checkpoint_best_pixel_aupimo_1e-5_1e-3.pt",
            "checkpoint_best_pixel_ap.pt",
            "checkpoint.pt",
        )
    )
    try:
        import mlflow

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or os.getenv("IQA_MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        model_version = client.get_model_version_by_alias(model_name, args.target_stage)
        run_id = str(getattr(model_version, "run_id", "") or "")
        if not run_id:
            raise ValueError(f"registry model {model_name!r} alias {args.target_stage!r} has no source run")
        registry_version = str(getattr(model_version, "version", "") or "")
        safe_model_name = model_name.replace("/", "_").replace("\\", "_").replace(":", "_")
        target_dir = (
            args.model_cache_root
            / args.scenario_id
            / "registry_initial"
            / role
            / f"{safe_model_name}_v{registry_version or 'unknown'}"
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path: Path | None = None
        last_error: Exception | None = None
        for checkpoint_name in checkpoint_names:
            try:
                downloaded = client.download_artifacts(
                    run_id,
                    f"{artifact_path}/artifacts/{checkpoint_name}",
                    str(target_dir),
                )
                checkpoint_path = Path(downloaded)
                break
            except Exception as exc:  # noqa: PERF203 - keep the last actionable MLflow error.
                last_error = exc
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"no checkpoint artifact found for {model_name} ({artifact_path}); last_error={last_error}"
            )
        validate_feature_ae_checkpoint_file(checkpoint_path, role=f"initial_{role}")
        run = client.get_run(run_id)
        run_tags = dict(getattr(run.data, "tags", {}) or {})
        version_name = (
            run_tags.get("candidate_version")
            or run_tags.get("feature_ae_version")
            or f"{model_name}:v{registry_version or 'unknown'}"
        )
        reference_contract = load_feature_ae_reference_contract(DEFAULT_FEATURE_AE_MODEL_VERSION)
        runtime_contract_path = _download_registered_runtime_contract(
            client,
            run_id=run_id,
            target_dir=target_dir,
            role=role,
            artifact_path=artifact_path,
        )
        registry_status = "loaded_from_registry_initial_without_runtime_contract"
        if runtime_contract_path is not None:
            runtime_contract = load_feature_ae_runtime_contract(runtime_contract_path)
            thresholds = dict(runtime_contract["decision_thresholds"])
            reference_contract = feature_ae_reference_contract_from_payload(
                runtime_contract.get("feature_ae_reference_contract") or runtime_contract["score_contract"]
            )
            registry_status = "loaded_from_registry_initial"
        else:
            try:
                thresholds = resolve_runtime_thresholds(version_name)
            except Exception:
                thresholds = dict(fallback_thresholds)
            try:
                reference_contract = load_feature_ae_reference_contract(version_name)
            except Exception:
                pass
        return ActiveRuntimeModel(
            version=version_name,
            checkpoint=checkpoint_path,
            decision_thresholds=thresholds,
            reference_contract=reference_contract,
            registry_model_name=model_name,
            registry_stage=args.target_stage,
            registry_alias=args.target_stage,
            registered_model_version=registry_version,
            registry_status=registry_status,
            registry_source_of_truth="mlflow_registry",
        )
    except Exception:
        if getattr(args, "require_mlflow_registry", False):
            raise
        return None


def _download_registered_runtime_contract(
    client: Any,
    *,
    run_id: str,
    target_dir: Path,
    role: str,
    artifact_path: str,
) -> Path | None:
    for candidate in (
        f"runtime_contracts/{role}/runtime_contract.json",
        f"{artifact_path}/artifacts/runtime_contract.json",
    ):
        try:
            downloaded = client.download_artifacts(run_id, candidate, str(target_dir))
        except Exception:
            continue
        path = Path(downloaded)
        if path.is_file():
            return path
    return None


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
    setattr(args, "lifecycle_run_id", state.run_id)

    roi_checkpoint = resolve_roi_segmenter_checkpoint(DEFAULT_ROI_MODEL_VERSION, strict_checksum=True)
    active_runtime = ActiveRuntimeModel(
        version=DEFAULT_FEATURE_AE_MODEL_VERSION,
        checkpoint=resolve_feature_ae_checkpoint(DEFAULT_FEATURE_AE_MODEL_VERSION, strict_checksum=True),
        decision_thresholds=resolve_runtime_thresholds(DEFAULT_FEATURE_AE_MODEL_VERSION),
        reference_contract=load_feature_ae_reference_contract(DEFAULT_FEATURE_AE_MODEL_VERSION),
        registry_model_name=registered_model_name(args.scenario_id),
        registry_stage=args.target_stage,
    )
    state.active_localization_runtime = active_runtime
    registry_classification_runtime = resolve_registered_initial_runtime(
        args,
        model_name=str(getattr(args, "initial_classification_registered_model", "") or ""),
        role="classification",
        fallback_thresholds=active_runtime.decision_thresholds,
    )
    if registry_classification_runtime is not None:
        active_runtime = registry_classification_runtime
    registry_localization_runtime = resolve_registered_initial_runtime(
        args,
        model_name=str(getattr(args, "initial_localization_registered_model", "") or ""),
        role="localization",
        fallback_thresholds=active_runtime.decision_thresholds,
    )
    state.active_localization_runtime = registry_localization_runtime or active_runtime
    state.active_model_initial = active_runtime.version
    state.active_model_final = active_runtime.version
    state.active_localization_model_initial = state.active_localization_runtime.version
    state.active_localization_model_final = state.active_localization_runtime.version
    state.promotion_chain = [active_runtime.version]
    state.classification_promotion_chain = [active_runtime.version]
    state.localization_promotion_chain = [state.active_localization_runtime.version]
    visual_store = create_visual_object_store()
    rows = load_replay_rows(args.scenario_id)
    if should_preflight_lifecycle_assets(args):
        preflight_lifecycle_assets(args, replay_rows=rows)
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
    emit_lifecycle_api_event(
        args,
        "run_started",
        candidate_initial_model_version=active_runtime.version,
        active_classification_model_version=active_runtime.version,
        active_localization_model_version=state.active_localization_runtime.version,
    )

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
                feature_reference_contract=active_runtime.reference_contract,
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
        if args.publish_minio and state.candidate_checkpoint:
            upload_checkpoint_to_s3(
                state.candidate_checkpoint,
                f"s3://iqa-models/{state.trigger_decision.candidate_dataset_version}/checkpoint.pt",
            )

    summary = summary_with_runtime(state, active_runtime, artifacts)
    write_json(artifacts.summary_path, summary)
    write_progress(artifacts, state, active_runtime, phase="completed")
    record_lifecycle_event(artifacts, state, active_runtime, "run_completed", summary=summary)
    emit_lifecycle_api_event(
        args,
        "run_completed",
        cycle_id=state.cycles[-1].get("cycle_id") if state.cycles else None,
        candidate_version=state.cycles[-1].get("candidate_version") if state.cycles else None,
        localization_promotion_status=state.cycles[-1].get("localization_promotion_status") if state.cycles else None,
        classification_promotion_status=state.cycles[-1].get("classification_promotion_status") if state.cycles else None,
    )
    return summary


def load_replay_rows(scenario_id: str) -> list[dict[str, str]]:
    plan = resolve_replay_plan(scenario_id)
    with plan.open(newline="", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row.get("scenario_id") == scenario_id]
    if not rows:
        raise ValueError(f"replay plan has no rows for scenario_id={scenario_id}: {plan}")
    return rows


def should_preflight_lifecycle_assets(args: argparse.Namespace) -> bool:
    return args.scenario_id == PIECE_B_TO_PIECE_A_P4_DRIFT_SCENARIO_ID and args.mode == "progressive-train"


def preflight_lifecycle_assets(args: argparse.Namespace, *, replay_rows: list[dict[str, str]]) -> None:
    image_root = Path(args.image_root)
    missing: list[str] = []
    replay_plan = resolve_replay_plan(args.scenario_id)
    missing.extend(
        validate_manifest_runtime_assets(
            replay_plan,
            image_root=image_root,
            rows=replay_rows,
            image_columns=("relative_paths", "relative_path"),
            mask_columns=("gt_mask_paths", "gt_mask_path", "mask_paths", "mask_path"),
        )
    )
    missing.extend(
        validate_manifest_runtime_assets(
            Path(args.reference_eval_manifest),
            image_root=image_root,
            image_columns=("relative_paths", "relative_path"),
            mask_columns=("gt_mask_paths", "gt_mask_path", "mask_paths", "mask_path"),
        )
    )
    missing.extend(
        validate_manifest_runtime_assets(
            Path(args.reference_gt_masks_manifest),
            image_root=image_root,
            mask_columns=("gt_mask_path", "mask_path", "path"),
        )
    )
    if missing:
        preview = "\n".join(f"- {item}" for item in missing[:20])
        extra = f"\n... {len(missing) - 20} more missing assets" if len(missing) > 20 else ""
        raise FileNotFoundError(
            "Lifecycle asset preflight failed. Publish and materialize the replay/evaluation manifests "
            f"before running the lifecycle. image_root={image_root}\n{preview}{extra}"
        )


def validate_manifest_runtime_assets(
    manifest_path: Path,
    *,
    image_root: Path,
    rows: list[dict[str, str]] | None = None,
    image_columns: tuple[str, ...] = (),
    mask_columns: tuple[str, ...] = (),
) -> list[str]:
    if rows is None:
        with manifest_path.open(newline="", encoding="utf-8-sig") as file:
            rows = list(csv.DictReader(file))
    missing: list[str] = []
    for row_index, row in enumerate(rows, start=2):
        for kind, columns in (("image", image_columns), ("gt_mask", mask_columns)):
            for value in manifest_path_values(row, columns):
                if not resolve_runtime_asset_path(value, image_root=image_root):
                    missing.append(f"{manifest_path}:{row_index}:{kind}:{value}")
    return missing


def manifest_path_values(row: dict[str, str], columns: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    for column in columns:
        raw_value = row.get(column) or ""
        values.extend(item for item in raw_value.replace(";", "|").replace(",", "|").split("|") if item.strip())
    return values


def resolve_runtime_asset_path(value: str, *, image_root: Path) -> Path | None:
    raw = Path(value.strip())
    if raw.is_absolute():
        return raw if raw.is_file() else None
    normalized = Path(normalize_runtime_asset_path(value))
    candidate = image_root / normalized
    return candidate if candidate.is_file() else None


def normalize_runtime_asset_path(value: str) -> str:
    path = value.strip().replace("\\", "/")
    for prefix in ("../raw/hss-iad/", "data/raw/hss-iad/", "raw/hss-iad/", "hss-iad/"):
        if path.startswith(prefix):
            return path[len(prefix) :]
    return path


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
    feature_reference_contract: Any | None = None,
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
        reference_contract=feature_reference_contract or load_feature_ae_reference_contract(DEFAULT_FEATURE_AE_MODEL_VERSION),
    )
    heatmap_uri = publish_heatmap(heatmap_path, context, store=visual_store) if heatmap_path.exists() else None
    return CycleEvent(
        event_id=event_id,
        piece_event_id=piece_event_id,
        lot_id=lot_id,
        scenario_id=scenario_id,
        scenario_phase=row.get("scenario_phase") or "",
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


def lifecycle_decision_for_lot(
    state: CycleState,
    lot: LotAccumulator,
    *,
    interval: int = 50,
    external_drift_confirmed: bool = False,
) -> LifecycleDecision:
    state.total_conforming_validated_count += lot.conforming_validated_count
    conforming_since_cycle = state.total_conforming_validated_count - state.last_cycle_conforming_validated_count
    piece_a_p4_correction_ready = (
        state.scenario_id == PIECE_B_TO_PIECE_A_P4_DRIFT_SCENARIO_ID
        and external_drift_confirmed
        and "correction_replay" in lot.scenario_phases
    )
    signal = LifecycleSignal(
        scenario_id=state.scenario_id,
        conforming_validated_count=conforming_since_cycle,
        drift_confirmed=state.scenario_id == DRIFT_SCENARIO_ID or piece_a_p4_correction_ready,
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
    decision = lifecycle_decision_for_lot(
        state,
        lot,
        interval=args.lifecycle_interval,
        external_drift_confirmed=bool(getattr(args, "external_drift_confirmed", False)),
    )
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
    emit_lifecycle_api_event(
        args,
        "cycle_started",
        cycle_id=cycle_id,
        candidate_version=f"rd_feature_ae_gated_natural_cycle_{cycle_number:03d}",
        candidate_initial_model_version=active_runtime.version,
        active_classification_model_version=active_runtime.version,
        active_localization_model_version=(
            state.active_localization_runtime.version
            if state.active_localization_runtime is not None
            else active_runtime.version
        ),
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
        )
    if args.dual_promotion and args.mode == "progressive-train":
        activation_started = datetime.now(UTC)
        promoted = str(cycle_result["candidate_version"])
        classification_promoted = cycle_result.get("classification_promotion_status") == "promoted"
        localization_promoted = cycle_result.get("localization_promotion_status") == "promoted"
        if classification_promoted:
            validate_feature_ae_checkpoint_file(
                Path(str(cycle_result.get("classification_candidate_checkpoint") or cycle_result["candidate_checkpoint"])),
                role="classification_active",
            )
            classification_thresholds = dict(
                cycle_result.get("classification_candidate_decision_thresholds")
                or cycle_result.get("candidate_decision_thresholds")
                or {}
            )
            state.active_model_final = promoted
            state.promotion_chain.append(promoted)
            state.classification_promotion_chain.append(promoted)
            state.candidate_checkpoint = str(cycle_result.get("classification_candidate_checkpoint") or cycle_result.get("candidate_checkpoint") or "")
            state.mlflow_run_id = str(cycle_result.get("mlflow_run_id") or "")
            active_runtime = ActiveRuntimeModel(
                version=promoted,
                checkpoint=Path(str(cycle_result.get("classification_candidate_checkpoint") or cycle_result["candidate_checkpoint"])),
                decision_thresholds=classification_thresholds,
                reference_contract=(
                    reference_contract_from_runtime_contract_path(cycle_result.get("classification_runtime_contract_path"))
                    or active_runtime.reference_contract
                ),
                registry_model_name=str(cycle_result.get("classification_registered_model_name") or registered_model_name(state.scenario_id, base_name=CLASSIFICATION_MODEL_NAME_BASE)),
                registry_stage=str(cycle_result.get("classification_registry_stage") or args.target_stage),
                registry_alias=str(cycle_result.get("classification_registry_alias") or args.target_stage),
                registered_model_version=str(cycle_result.get("classification_registered_model_version") or ""),
                registry_status=str(cycle_result.get("classification_registry_status") or ""),
                registry_source_of_truth=str(cycle_result.get("classification_registry_source_of_truth") or ""),
            )
        if localization_promoted:
            validate_feature_ae_checkpoint_file(
                Path(str(cycle_result.get("localization_candidate_checkpoint") or cycle_result["candidate_checkpoint"])),
                role="localization_active",
            )
            localization_thresholds = dict(
                cycle_result.get("localization_candidate_decision_thresholds")
                or cycle_result.get("candidate_decision_thresholds")
                or {}
            )
            state.active_localization_model_final = promoted
            state.localization_promotion_chain.append(promoted)
            state.active_localization_runtime = ActiveRuntimeModel(
                version=promoted,
                checkpoint=Path(str(cycle_result.get("localization_candidate_checkpoint") or cycle_result["candidate_checkpoint"])),
                decision_thresholds=localization_thresholds,
                reference_contract=(
                    reference_contract_from_runtime_contract_path(cycle_result.get("localization_runtime_contract_path"))
                    or (
                        state.active_localization_runtime.reference_contract
                        if state.active_localization_runtime is not None
                        else active_runtime.reference_contract
                    )
                ),
                registry_model_name=str(cycle_result.get("localization_registered_model_name") or registered_model_name(state.scenario_id, base_name=LOCALIZATION_MODEL_NAME_BASE)),
                registry_stage=str(cycle_result.get("localization_registry_stage") or args.target_stage),
                registry_alias=str(cycle_result.get("localization_registry_alias") or args.target_stage),
                registered_model_version=str(cycle_result.get("localization_registered_model_version") or ""),
                registry_status=str(cycle_result.get("localization_registry_status") or ""),
                registry_source_of_truth=str(cycle_result.get("localization_registry_source_of_truth") or ""),
            )
        state.status = "trained" if (classification_promoted or localization_promoted) else "rejected"
        cycle_result["activated_for_next_events"] = classification_promoted or localization_promoted
        cycle_result["decision_model_activated_for_next_events"] = classification_promoted
        cycle_result["classification_activated_for_next_events"] = classification_promoted
        cycle_result["localization_activated_for_next_events"] = localization_promoted
        cycle_result["activation_event_index"] = state.events_processed
        cycle_result["activation_scope"] = "dual_promotion" if (classification_promoted or localization_promoted) else "unchanged"
        cycle_result["active_thresholds_after"] = active_runtime.decision_thresholds
        cycle_result["active_runtime_after"] = active_runtime.to_dict()
        cycle_result["active_classification_runtime_after"] = active_runtime.to_dict()
        cycle_result["active_localization_runtime_after"] = (
            state.active_localization_runtime.to_dict()
            if state.active_localization_runtime is not None
            else active_runtime.to_dict()
        )
        record_lifecycle_event(
            artifacts,
            state,
            active_runtime,
            "dual_gate_decision",
            cycle_id=cycle_id,
            candidate_version=promoted,
            localization_promotion_status=cycle_result.get("localization_promotion_status"),
            classification_promotion_status=cycle_result.get("classification_promotion_status"),
            localization_gate=cycle_result.get("localization_gate"),
            classification_gate=cycle_result.get("classification_gate"),
        )
        if classification_promoted:
            record_lifecycle_event(
                artifacts,
                state,
                active_runtime,
                "classification_model_activated",
                cycle_id=cycle_id,
                active_model_version=active_runtime.version,
                active_thresholds=active_runtime.decision_thresholds,
                activation_scope=cycle_result["activation_scope"],
            )
        if localization_promoted:
            record_lifecycle_event(
                artifacts,
                state,
                active_runtime,
                "localization_model_activated",
                cycle_id=cycle_id,
                active_model_version=state.active_localization_runtime.version if state.active_localization_runtime else promoted,
                activation_scope=cycle_result["activation_scope"],
            )
        record_timing(
            artifacts,
            "activation",
            duration_seconds=(datetime.now(UTC) - activation_started).total_seconds(),
            cycle_id=cycle_id,
            active_model_version=active_runtime.version,
        )
    elif cycle_result.get("promotion_status") == "promoted":
        activation_started = datetime.now(UTC)
        promoted = str(cycle_result["candidate_version"])
        validate_feature_ae_checkpoint_file(Path(str(cycle_result["candidate_checkpoint"])), role="active")
        state.active_model_final = promoted
        state.promotion_chain.append(promoted)
        state.candidate_checkpoint = str(cycle_result.get("candidate_checkpoint") or "")
        state.mlflow_run_id = str(cycle_result.get("mlflow_run_id") or "")
        state.status = "trained" if args.mode == "progressive-train" else "validated"
        active_runtime = ActiveRuntimeModel(
            version=promoted,
            checkpoint=Path(str(cycle_result["candidate_checkpoint"])),
            decision_thresholds=dict(cycle_result["candidate_decision_thresholds"]),
            reference_contract=(
                reference_contract_from_runtime_contract_path(
                    cycle_result.get("classification_runtime_contract_path") or cycle_result.get("runtime_contract_path")
                )
                or active_runtime.reference_contract
            ),
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
    emit_lifecycle_api_event(
        args,
        "promotion_decision",
        cycle_id=cycle_id,
        candidate_version=cycle_result.get("candidate_version"),
        candidate_initial_model_version=cycle_result.get("candidate_initial_model_version"),
        candidate_initial_checkpoint_sha256=cycle_result.get("candidate_initial_checkpoint_sha256"),
        localization_checkpoint_sha256=cycle_result.get("localization_checkpoint_sha256"),
        classification_checkpoint_sha256=cycle_result.get("classification_checkpoint_sha256"),
        localization_promotion_status=cycle_result.get("localization_promotion_status"),
        classification_promotion_status=cycle_result.get("classification_promotion_status"),
        localization_gate_reason=cycle_result.get("localization_gate_reason"),
        classification_gate_reason=cycle_result.get("classification_gate_reason"),
        metrics={
            "localization_metric_delta": cycle_result.get("localization_metric_delta"),
            "classification_metric_delta": cycle_result.get("classification_metric_delta"),
            "classification_fn_delta": cycle_result.get("fn_delta"),
        },
    )
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
    reference_evaluation_set_id = args.reference_eval_manifest.stem
    active_model_before = active_runtime.version
    active_checkpoint = active_runtime.checkpoint
    active_localization_runtime = state.active_localization_runtime or active_runtime
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
    reference_evaluation_set_path = cycle_dir / "reference_evaluation_set.csv"
    shutil.copy2(args.reference_eval_manifest, reference_evaluation_set_path)
    shutil.copy2(reference_evaluation_set_path, cycle_dir / "evaluation_set.csv")
    result: dict[str, Any] = {
        "cycle_id": f"cycle_{cycle_number:03d}",
        "run_id": state.run_id,
        "lifecycle_run_id": state.run_id,
        "scenario_id": state.scenario_id,
        "promotion_policy": PROGRESSIVE_PROMOTION_POLICY,
        "dual_promotion": bool(args.dual_promotion),
        "promotion_min_delta": float(args.promotion_min_delta),
        "localization_promotion_min_delta": float(args.localization_promotion_min_delta),
        "active_model_before": active_model_before,
        "classification_model_before": active_model_before,
        "localization_model_before": active_localization_runtime.version,
        "active_runtime_before": active_runtime.to_dict(),
        "active_classification_runtime_before": active_runtime.to_dict(),
        "active_localization_runtime_before": active_localization_runtime.to_dict(),
        "active_thresholds_before": active_runtime.decision_thresholds,
        "candidate_version": candidate_version,
        "candidate_run_dir": str(progressive_candidate_run_dir(args, candidate_version)),
        "dataset_snapshot_id": dataset_snapshot_id,
        "dataset_snapshot_path": str(seen_snapshot_path),
        "training_manifest_path": str(training_manifest_path),
        "training_manifest_stats": training_manifest_stats,
        "candidate_init_policy": args.candidate_init_policy,
        "calibration_set_id": calibration_set_id,
        "evaluation_set_id": reference_evaluation_set_id,
        "evaluation_set_path": str(reference_evaluation_set_path),
        "reference_evaluation_set_id": reference_evaluation_set_id,
        "reference_evaluation_set_path": str(reference_evaluation_set_path),
        "classification_selection_manifest": str(getattr(args, "classification_selection_manifest", "") or ""),
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
        "localization_registry_status": "not_registered",
        "classification_registry_status": "not_registered",
        "localization_promotion_status": "simulated",
        "classification_promotion_status": "simulated",
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
        initial_checkpoint_reference = safe_checkpoint_reference(initial_checkpoint_path)
        initial_checkpoint_sha256 = sha256_file(initial_checkpoint_path) if initial_checkpoint_path else ""
        initial_model_version = resolve_candidate_initial_model_version(args, active_runtime=active_runtime)
        train_result = train_progressive_candidate(
            args,
            candidate_version,
            training_manifest_path,
            dataset_snapshot_id,
            initial_checkpoint_path=initial_checkpoint_path,
            initial_checkpoint_reference=initial_checkpoint_reference,
            initial_checkpoint_sha256=initial_checkpoint_sha256,
            initial_model_version=initial_model_version,
            cycle_id=f"cycle_{cycle_number:03d}",
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
        specialized = resolve_specialized_candidate_checkpoints(train_result)
        classification_selection = select_classification_candidate_checkpoint(
            args,
            cycle_dir=cycle_dir,
            candidate_version=candidate_version,
            default_checkpoint=specialized["classification"],
            active_runtime=active_runtime,
            artifacts=artifacts,
            state=state,
        )
        if classification_selection.get("selected_checkpoint"):
            specialized["classification"] = Path(str(classification_selection["selected_checkpoint"]))
        result["classification_checkpoint_selection"] = classification_selection
        result["classification_selection_manifest"] = classification_selection.get("selection_manifest") or result.get("classification_selection_manifest")
        result["classification_selection_reason"] = classification_selection.get("reason")
        result["classification_selection_checkpoint_sha256"] = classification_selection.get("selected_checkpoint_sha256")
        result["classification_selection_false_negatives"] = classification_selection.get("selected_false_negatives")
        result["classification_selection_image_recall"] = classification_selection.get("selected_image_recall")
        result["classification_selection_image_ap"] = classification_selection.get("selected_image_ap")
        result["localization_candidate_checkpoint"] = str(specialized["localization"])
        result["classification_candidate_checkpoint"] = str(specialized["classification"])
        checkpoint_evidence = validate_cycle_checkpoints(
            candidate_checkpoint=Path(str(result["candidate_checkpoint"])),
            localization_checkpoint=specialized["localization"],
            classification_checkpoint=specialized["classification"],
        )
        result.update(checkpoint_evidence)
        result["candidate_initial_checkpoint"] = initial_checkpoint_reference
        result["candidate_initial_checkpoint_sha256"] = initial_checkpoint_sha256
        result["candidate_initial_model_version"] = initial_model_version
        result["mlflow_run_id"] = str(train_result.get("run_id") or "")
        result["mlflow_dataset_logged"] = bool(train_result.get("mlflow_dataset_logged"))
        result["mlflow_training_dataset_logged"] = bool(train_result.get("mlflow_training_dataset_logged"))
        result["mlflow_metric_eval_dataset_logged"] = bool(train_result.get("mlflow_metric_eval_dataset_logged"))
        result["mlflow_model_logged"] = bool(train_result.get("mlflow_model_logged"))
        result["mlflow_localization_model_logged"] = bool(train_result.get("mlflow_localization_model_logged"))
        result["mlflow_classification_model_logged"] = bool(train_result.get("mlflow_classification_model_logged"))
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
                "candidate_run_dir": result.get("candidate_run_dir"),
                "candidate_initial_model_version": result.get("candidate_initial_model_version"),
                "candidate_initial_checkpoint": result.get("candidate_initial_checkpoint"),
                "candidate_initial_checkpoint_sha256": result.get("candidate_initial_checkpoint_sha256"),
                "localization_candidate_checkpoint": result.get("localization_candidate_checkpoint"),
                "classification_candidate_checkpoint": result.get("classification_candidate_checkpoint"),
                "localization_checkpoint_sha256": result.get("localization_checkpoint_sha256"),
                "classification_checkpoint_sha256": result.get("classification_checkpoint_sha256"),
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
            candidate_run_dir=result.get("candidate_run_dir"),
            candidate_initial_model_version=result.get("candidate_initial_model_version"),
            candidate_initial_checkpoint=result.get("candidate_initial_checkpoint"),
            candidate_initial_checkpoint_sha256=result.get("candidate_initial_checkpoint_sha256"),
            localization_candidate_checkpoint=result.get("localization_candidate_checkpoint"),
            classification_candidate_checkpoint=result.get("classification_candidate_checkpoint"),
            localization_checkpoint_sha256=result.get("localization_checkpoint_sha256"),
            classification_checkpoint_sha256=result.get("classification_checkpoint_sha256"),
        )
        comparison = evaluate_reference_promotion_comparison(
            args,
            cycle_dir=cycle_dir,
            reference_evaluation_set_path=reference_evaluation_set_path,
            reference_evaluation_set_id=reference_evaluation_set_id,
            active_model_version=active_model_before,
            active_checkpoint_path=Path(active_checkpoint),
            active_localization_runtime=active_localization_runtime,
            candidate_version=candidate_version,
            candidate_checkpoint_path=Path(str(result["candidate_checkpoint"])),
            localization_candidate_checkpoint_path=Path(str(result["localization_candidate_checkpoint"])),
            classification_candidate_checkpoint_path=Path(str(result["classification_candidate_checkpoint"])),
            artifacts=artifacts,
            state=state,
            active_runtime=active_runtime,
        )
        result.update(comparison)
        result["promotion_status"] = comparison["promotion_status"]
        result["gate_decision"] = comparison["gate_decision"]
        result["gate_reason"] = comparison["gate_reason"]
        record_lifecycle_event(
            artifacts,
            state,
            active_runtime,
            "gate_decision",
            cycle_id=f"cycle_{cycle_number:03d}",
            candidate_version=candidate_version,
            gate_eval_profile=args.gate_eval_profile,
            gate_decision=result.get("gate_decision"),
            gate_reason=result.get("gate_reason"),
            promotion_status=result.get("promotion_status"),
            selected_metric=result.get("selected_metric"),
            metric_delta=result.get("metric_delta"),
            active_false_negatives=result.get("active_false_negatives"),
            candidate_false_negatives=result.get("candidate_false_negatives"),
            active_good_red_count=result.get("active_good_red_count"),
            candidate_good_red_count=result.get("candidate_good_red_count"),
        )
        emit_lifecycle_api_event(
            args,
            "gate_decision",
            cycle_id=f"cycle_{cycle_number:03d}",
            candidate_version=candidate_version,
            candidate_initial_model_version=result.get("candidate_initial_model_version"),
            candidate_initial_checkpoint_sha256=result.get("candidate_initial_checkpoint_sha256"),
            localization_checkpoint_sha256=result.get("localization_checkpoint_sha256"),
            classification_checkpoint_sha256=result.get("classification_checkpoint_sha256"),
            localization_promotion_status=result.get("localization_promotion_status"),
            classification_promotion_status=result.get("classification_promotion_status"),
            localization_gate_reason=result.get("localization_gate_reason"),
            classification_gate_reason=result.get("classification_gate_reason"),
            metrics={
                "localization_metric_delta": result.get("localization_metric_delta"),
                "classification_metric_delta": result.get("classification_metric_delta"),
                "classification_fn_delta": result.get("fn_delta"),
            },
        )
        publish_promoted_checkpoints(args, state, result)
        if args.dual_promotion:
            if result.get("localization_promotion_status") == "promoted":
                result.update(prefix_registry_result("localization", register_promoted_cycle(
                    args,
                    state,
                    result,
                    model_name_base=LOCALIZATION_MODEL_NAME_BASE,
                    model_role="localization",
                )))
                tag_mlflow_promotion_evidence(role_promotion_cycle(result, "localization"))
            if result.get("classification_promotion_status") == "promoted":
                result.update(prefix_registry_result("classification", register_promoted_cycle(
                    args,
                    state,
                    result,
                    model_name_base=CLASSIFICATION_MODEL_NAME_BASE,
                    model_role="classification",
                )))
                tag_mlflow_promotion_evidence(role_promotion_cycle(result, "classification"))
            tag_mlflow_promotion_evidence(result)
        elif result["promotion_status"] == "promoted":
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
        if not args.dual_promotion:
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
    history_path = next((path.parent / "epoch_metrics.jsonl" for path in candidates if (path.parent / "epoch_metrics.jsonl").is_file()), None)
    if history_path:
        epoch_metric_history = [
            json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
    else:
        epoch_metric_history = train_result.get("epoch_metric_history", [])
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
    metric_priority: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    started_at = datetime.now(UTC)
    cache_root = DEFAULT_METRIC_CACHE_ROOT
    record_lifecycle_event(
        artifacts,
        state,
        active_runtime,
        "active_eval_start",
        cycle_id=cycle_dir.name,
        panel=panel_name,
        gate_eval_profile=args.gate_eval_profile,
        model_version=active_model_version,
    )
    active = evaluate_reference_model_on_set(
        args,
        model_version=active_model_version,
        checkpoint_path=active_checkpoint_path,
        evaluation_set_path=evaluation_set_path,
        output_dir=cycle_dir / "evaluation" / panel_name / "active_before",
        evaluation_set_id=evaluation_set_id,
        cache_root=cache_root,
        gt_masks_manifest=gt_masks_manifest,
        cache_enabled=True,
    )
    record_lifecycle_event(
        artifacts,
        state,
        active_runtime,
        "active_eval_cache_hit" if active.get("cache_hit") else "active_eval_cache_miss",
        cycle_id=cycle_dir.name,
        panel=panel_name,
        gate_eval_profile=args.gate_eval_profile,
        cache_status=active.get("cache_status"),
        cache_key=active.get("cache_key"),
    )
    record_lifecycle_event(
        artifacts,
        state,
        active_runtime,
        "candidate_eval_start",
        cycle_id=cycle_dir.name,
        panel=panel_name,
        gate_eval_profile=args.gate_eval_profile,
        model_version=candidate_version,
    )
    candidate = evaluate_reference_model_on_set(
        args,
        model_version=candidate_version,
        checkpoint_path=candidate_checkpoint_path,
        evaluation_set_path=evaluation_set_path,
        output_dir=cycle_dir / "evaluation" / panel_name / "candidate",
        evaluation_set_id=evaluation_set_id,
        cache_root=cache_root,
        gt_masks_manifest=gt_masks_manifest,
        cache_enabled=True,
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
    if active.get("cache_status") == "miss":
        store_metric_cache(cache_root, str(active.get("cache_key")), cycle_dir / "evaluation" / panel_name / "active_before")
        active["cache_status"] = "miss_stored"
    record_lifecycle_event(
        artifacts,
        state,
        active_runtime,
        "active_eval_done",
        cycle_id=cycle_dir.name,
        panel=panel_name,
        gate_eval_profile=args.gate_eval_profile,
        cache_status=active.get("cache_status"),
        image_count=active.get("image_count"),
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
    if candidate.get("cache_status") == "miss":
        store_metric_cache(cache_root, str(candidate.get("cache_key")), cycle_dir / "evaluation" / panel_name / "candidate")
        candidate["cache_status"] = "miss_stored"
    record_lifecycle_event(
        artifacts,
        state,
        active_runtime,
        "candidate_eval_done",
        cycle_id=cycle_dir.name,
        panel=panel_name,
        gate_eval_profile=args.gate_eval_profile,
        cache_status=candidate.get("cache_status"),
        image_count=candidate.get("image_count"),
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
                gate_eval_profile=args.gate_eval_profile,
                image_count=payload.get("image_count"),
                metric_timings=payload.get("metric_timings") or {},
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
                gate_eval_profile=args.gate_eval_profile,
                image_count=payload.get("image_count"),
                metric_timings=metric_timings,
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
                gate_eval_profile=args.gate_eval_profile,
                image_count=payload.get("image_count"),
                metric_timings=metric_timings,
            )
    record_timing(
        artifacts,
        "comparative_eval",
        duration_seconds=duration,
        cycle_id=cycle_dir.name,
        panel=panel_name,
        gate_eval_profile=args.gate_eval_profile,
        active_cache_status=active.get("cache_status"),
        candidate_cache_status=candidate.get("cache_status"),
        active_image_count=active.get("image_count"),
        candidate_image_count=candidate.get("image_count"),
        active_metric_timings=active.get("metric_timings"),
        candidate_metric_timings=candidate.get("metric_timings"),
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
    selected_metric = select_comparable_business_metric(active["metrics"], candidate["metrics"], priority=metric_priority)
    record_lifecycle_event(
        artifacts,
        state,
        active_runtime,
        "gate_metrics_computed",
        cycle_id=cycle_dir.name,
        panel=panel_name,
        gate_eval_profile=args.gate_eval_profile,
        selected_metric=selected_metric,
        active_cache_status=active.get("cache_status"),
        candidate_cache_status=candidate.get("cache_status"),
    )
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
            "gate_eval_profile": args.gate_eval_profile,
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
        "evaluation_started_at": started_at.isoformat(),
        "evaluation_completed_at": completed_at.isoformat(),
        "evaluation_duration_seconds": (completed_at - started_at).total_seconds(),
        "operational_alerts": operational_alerts,
        "defective_count": count_defective_rows(evaluation_set_path),
    }


def evaluate_dual_reference_promotion_comparison(
    args: argparse.Namespace,
    *,
    cycle_dir: Path,
    reference_evaluation_set_path: Path,
    reference_evaluation_set_id: str,
    active_model_version: str,
    active_checkpoint_path: Path,
    active_localization_runtime: ActiveRuntimeModel,
    candidate_version: str,
    localization_candidate_checkpoint_path: Path,
    classification_candidate_checkpoint_path: Path,
    artifacts: LifecycleArtifacts,
    state: CycleState,
    active_runtime: ActiveRuntimeModel,
    started_at: datetime,
) -> dict[str, Any]:
    promotion_objective = PROGRESSIVE_PROMOTION_POLICY
    piece_b_non_regression_policy = PIECE_B_NON_REGRESSION_BLOCKING
    gate_evaluation_set_path = reference_evaluation_set_path
    gate_evaluation_set_id = reference_evaluation_set_id
    gate_panel_prefix = "reference"
    report_reference_localization: dict[str, Any] | None = None
    report_reference_classification: dict[str, Any] | None = None
    classification_selection_manifest = getattr(args, "classification_selection_manifest", None)
    classification_selection_path = Path(str(classification_selection_manifest)) if classification_selection_manifest else None
    if (
        args.scenario_id == PIECE_B_TO_PIECE_A_P4_DRIFT_SCENARIO_ID
        and classification_selection_path is not None
        and classification_selection_path.is_file()
    ):
        promotion_objective = PIECE_A_P4_DEMO_PROMOTION_OBJECTIVE
        piece_b_non_regression_policy = PIECE_B_NON_REGRESSION_REPORT_ONLY
        gate_evaluation_set_path = classification_selection_path
        gate_evaluation_set_id = classification_selection_path.stem
        gate_panel_prefix = "piece_a_p4_correction"

    localization = evaluate_model_pair_on_panel(
        args,
        cycle_dir=cycle_dir,
        panel_name=f"{gate_panel_prefix}_localization",
        evaluation_set_path=gate_evaluation_set_path,
        evaluation_set_id=gate_evaluation_set_id,
        active_model_version=active_localization_runtime.version,
        active_checkpoint_path=active_localization_runtime.checkpoint,
        candidate_version=candidate_version,
        candidate_checkpoint_path=localization_candidate_checkpoint_path,
        artifacts=artifacts,
        state=state,
        active_runtime=active_runtime,
        active_decision_thresholds=active_localization_runtime.decision_thresholds,
        gt_masks_manifest=args.reference_gt_masks_manifest,
        metric_priority=LOCALIZATION_METRIC_PRIORITY,
    )
    classification = evaluate_model_pair_on_panel(
        args,
        cycle_dir=cycle_dir,
        panel_name=f"{gate_panel_prefix}_classification",
        evaluation_set_path=gate_evaluation_set_path,
        evaluation_set_id=gate_evaluation_set_id,
        active_model_version=active_model_version,
        active_checkpoint_path=active_checkpoint_path,
        candidate_version=candidate_version,
        candidate_checkpoint_path=classification_candidate_checkpoint_path,
        artifacts=artifacts,
        state=state,
        active_runtime=active_runtime,
        active_decision_thresholds=active_runtime.decision_thresholds,
        gt_masks_manifest=args.reference_gt_masks_manifest,
        metric_priority=CLASSIFICATION_METRIC_PRIORITY,
    )
    if piece_b_non_regression_policy == PIECE_B_NON_REGRESSION_REPORT_ONLY:
        report_reference_localization = evaluate_model_pair_on_panel(
            args,
            cycle_dir=cycle_dir,
            panel_name="report_piece_b_non_regression_localization",
            evaluation_set_path=reference_evaluation_set_path,
            evaluation_set_id=reference_evaluation_set_id,
            active_model_version=active_localization_runtime.version,
            active_checkpoint_path=active_localization_runtime.checkpoint,
            candidate_version=candidate_version,
            candidate_checkpoint_path=localization_candidate_checkpoint_path,
            artifacts=artifacts,
            state=state,
            active_runtime=active_runtime,
            active_decision_thresholds=active_localization_runtime.decision_thresholds,
            gt_masks_manifest=args.reference_gt_masks_manifest,
            metric_priority=LOCALIZATION_METRIC_PRIORITY,
        )
        report_reference_classification = evaluate_model_pair_on_panel(
            args,
            cycle_dir=cycle_dir,
            panel_name="report_piece_b_non_regression_classification",
            evaluation_set_path=reference_evaluation_set_path,
            evaluation_set_id=reference_evaluation_set_id,
            active_model_version=active_model_version,
            active_checkpoint_path=active_checkpoint_path,
            candidate_version=candidate_version,
            candidate_checkpoint_path=classification_candidate_checkpoint_path,
            artifacts=artifacts,
            state=state,
            active_runtime=active_runtime,
            active_decision_thresholds=active_runtime.decision_thresholds,
            gt_masks_manifest=args.reference_gt_masks_manifest,
            metric_priority=CLASSIFICATION_METRIC_PRIORITY,
        )
    completed_at = datetime.now(UTC)

    localization_delta = _finite_float(localization.get("metric_delta"))
    localization_metric_ok = (
        localization.get("selected_metric") is not None
        and localization_delta is not None
        and localization_delta > float(args.localization_promotion_min_delta)
    )
    localization_gate = {
        "metric": localization.get("selected_metric"),
        "active_value": localization.get("active_metric_value"),
        "candidate_value": localization.get("candidate_metric_value"),
        "delta": localization.get("metric_delta"),
        "min_delta": float(args.localization_promotion_min_delta),
        "passed": bool(localization_metric_ok),
    }

    active_metrics = classification.get("active_metrics_on_eval_set") or {}
    candidate_metrics = classification.get("candidate_metrics_on_eval_set") or {}
    active_fn = int(active_metrics.get("false_negatives") or 0)
    candidate_fn = int(candidate_metrics.get("false_negatives") or 0)
    fn_delta = candidate_fn - active_fn
    active_good_red_count = int(active_metrics.get("good_red_count") or 0)
    candidate_good_red_count = int(candidate_metrics.get("good_red_count") or 0)
    good_red_delta = candidate_good_red_count - active_good_red_count
    active_image_recall = _finite_float(active_metrics.get("image_recall"))
    candidate_image_recall = _finite_float(candidate_metrics.get("image_recall"))
    image_recall_delta = (
        None
        if active_image_recall is None or candidate_image_recall is None
        else candidate_image_recall - active_image_recall
    )
    active_image_ap = _finite_float(active_metrics.get("image_ap"))
    candidate_image_ap = _finite_float(candidate_metrics.get("image_ap"))
    image_ap_delta = None if active_image_ap is None or candidate_image_ap is None else candidate_image_ap - active_image_ap
    good_red_ok = good_red_delta <= int(args.max_good_red_regression)
    fn_improved = fn_delta < 0
    fn_non_regression = fn_delta <= 0
    recall_improved = image_recall_delta is not None and image_recall_delta > float(args.classification_min_image_recall_delta)
    ap_improved = image_ap_delta is not None and image_ap_delta > float(args.classification_min_image_ap_delta)
    if promotion_objective == PIECE_A_P4_DEMO_PROMOTION_OBJECTIVE:
        classification_metric_ok = bool(fn_non_regression or recall_improved or ap_improved)
    else:
        classification_metric_ok = fn_improved if args.classification_require_fn_improvement else bool(fn_improved or recall_improved or ap_improved)
    thresholds_ok = classification.get("candidate_decision_thresholds") is not None and classification.get("active_decision_thresholds") is not None
    classification_passed = bool(classification_metric_ok and good_red_ok and thresholds_ok)
    classification_selected_metric = "false_negatives" if fn_improved else ("image_recall" if recall_improved else ("image_ap" if ap_improved else classification.get("selected_metric")))
    classification_gate = {
        "promotion_objective": promotion_objective,
        "piece_b_non_regression_policy": piece_b_non_regression_policy,
        "gate_evaluation_set_id": gate_evaluation_set_id,
        "selected_metric": classification_selected_metric,
        "active_false_negatives": active_fn,
        "candidate_false_negatives": candidate_fn,
        "fn_delta": fn_delta,
        "fn_improved": fn_improved,
        "fn_non_regression": fn_non_regression,
        "require_fn_improvement": bool(args.classification_require_fn_improvement),
        "active_image_recall": active_image_recall,
        "candidate_image_recall": candidate_image_recall,
        "image_recall_delta": image_recall_delta,
        "min_image_recall_delta": float(args.classification_min_image_recall_delta),
        "active_image_ap": active_image_ap,
        "candidate_image_ap": candidate_image_ap,
        "image_ap_delta": image_ap_delta,
        "min_image_ap_delta": float(args.classification_min_image_ap_delta),
        "active_good_red_count": active_good_red_count,
        "candidate_good_red_count": candidate_good_red_count,
        "good_red_delta": good_red_delta,
        "max_good_red_regression": int(args.max_good_red_regression),
        "good_red_ok": good_red_ok,
        "thresholds_ok": thresholds_ok,
        "metric_ok": classification_metric_ok,
        "passed": classification_passed,
    }
    classification_progress = _classification_progress_summary(
        classification_gate,
        reference_false_negatives_ok=fn_delta <= 0,
        good_red_ok=good_red_ok,
    )
    localization_status = "promoted" if localization_gate["passed"] else _dual_rejection_status(localization_gate, role="localization")
    classification_status = "promoted" if classification_passed else _dual_rejection_status(classification_gate, role="classification")
    localization_gate_reason = f"localization_{localization_status}"
    classification_gate_reason = f"classification_{classification_status}"
    localization_promoted = localization_status == "promoted"
    classification_promoted = classification_status == "promoted"
    top_level_status = "promoted" if classification_promoted else ("partially_promoted" if localization_promoted else classification_status)
    top_level_gate = "passed" if classification_promoted else ("partially_passed" if localization_promoted else "rejected")
    dual_promotion_outcome = f"localization_{localization_status}__classification_{classification_status}"
    localization_runtime_contract_path = write_feature_ae_runtime_contract(
        output_path=cycle_dir / "runtime_contracts" / "localization" / "runtime_contract.json",
        metrics_path=localization.get("candidate_eval_metrics_path"),
        checkpoint_path=localization_candidate_checkpoint_path,
        model_version=candidate_version,
        model_role="localization",
        decision_thresholds=localization.get("candidate_decision_thresholds"),
        scenario_id=args.scenario_id,
        lifecycle_run_id=str(getattr(args, "lifecycle_run_id", "")),
        cycle_id=cycle_dir.name,
    )
    classification_runtime_contract_path = write_feature_ae_runtime_contract(
        output_path=cycle_dir / "runtime_contracts" / "classification" / "runtime_contract.json",
        metrics_path=classification.get("candidate_eval_metrics_path"),
        checkpoint_path=classification_candidate_checkpoint_path,
        model_version=candidate_version,
        model_role="classification",
        decision_thresholds=classification.get("candidate_decision_thresholds"),
        scenario_id=args.scenario_id,
        lifecycle_run_id=str(getattr(args, "lifecycle_run_id", "")),
        cycle_id=cycle_dir.name,
    )

    return {
        "promotion_objective": promotion_objective,
        "piece_b_non_regression_policy": piece_b_non_regression_policy,
        "gate_evaluation_set_id": gate_evaluation_set_id,
        "gate_evaluation_set_path": str(gate_evaluation_set_path),
        "reference_evaluation_report_set_id": reference_evaluation_set_id,
        "reference_evaluation_report_set_path": str(reference_evaluation_set_path),
        "metrics": classification.get("metrics") or {},
        "active_metrics_on_eval_set": classification.get("active_metrics_on_eval_set") or {},
        "candidate_metrics_on_eval_set": classification.get("candidate_metrics_on_eval_set") or {},
        "localization_active_metrics_on_eval_set": localization.get("active_metrics_on_eval_set") or {},
        "localization_candidate_metrics_on_eval_set": localization.get("candidate_metrics_on_eval_set") or {},
        "classification_active_metrics_on_eval_set": classification.get("active_metrics_on_eval_set") or {},
        "classification_candidate_metrics_on_eval_set": classification.get("candidate_metrics_on_eval_set") or {},
        "selected_metric": classification_selected_metric,
        "selected_metric_value": candidate_metrics.get(classification_selected_metric) if classification_selected_metric else None,
        "active_metric_value": active_metrics.get(classification_selected_metric) if classification_selected_metric else None,
        "candidate_metric_value": candidate_metrics.get(classification_selected_metric) if classification_selected_metric else None,
        "metric_delta": -fn_delta if classification_selected_metric == "false_negatives" else classification.get("metric_delta"),
        "localization_selected_metric": localization.get("selected_metric"),
        "localization_active_metric_value": localization.get("active_metric_value"),
        "localization_candidate_metric_value": localization.get("candidate_metric_value"),
        "localization_metric_delta": localization.get("metric_delta"),
        "classification_selected_metric": classification_selected_metric,
        "classification_active_metric_value": active_metrics.get(classification_selected_metric) if classification_selected_metric else None,
        "classification_candidate_metric_value": candidate_metrics.get(classification_selected_metric) if classification_selected_metric else None,
        "classification_metric_delta": -fn_delta if classification_selected_metric == "false_negatives" else classification.get("metric_delta"),
        "gate_decision": top_level_gate,
        "gate_reason": "dual_promotion_evaluated",
        "promotion_status": top_level_status,
        "dual_promotion_outcome": dual_promotion_outcome,
        "localization_promotion_status": localization_status,
        "classification_promotion_status": classification_status,
        "localization_gate_reason": localization_gate_reason,
        "classification_gate_reason": classification_gate_reason,
        "localization_gate": localization_gate,
        "classification_gate": classification_gate,
        "classification_progress": classification_progress,
        "classification_progress_improved": classification_progress["improved"],
        "classification_progress_non_regression": classification_progress["non_regression"],
        "classification_progress_summary": classification_progress["summary"],
        "gate_eval_profile": args.gate_eval_profile,
        "active_decision_thresholds": classification.get("active_decision_thresholds"),
        "candidate_decision_thresholds": classification.get("candidate_decision_thresholds"),
        "localization_candidate_decision_thresholds": localization.get("candidate_decision_thresholds"),
        "classification_candidate_decision_thresholds": classification.get("candidate_decision_thresholds"),
        "threshold_source": (
            classification.get("candidate_decision_thresholds", {}) or {}
        ).get("threshold_source", ""),
        "active_eval_metrics_path": classification.get("active_eval_metrics_path"),
        "candidate_eval_metrics_path": classification.get("candidate_eval_metrics_path"),
        "localization_active_eval_metrics_path": localization.get("active_eval_metrics_path"),
        "localization_candidate_eval_metrics_path": localization.get("candidate_eval_metrics_path"),
        "classification_active_eval_metrics_path": classification.get("active_eval_metrics_path"),
        "classification_candidate_eval_metrics_path": classification.get("candidate_eval_metrics_path"),
        "piece_b_non_regression_localization_report": report_reference_localization or {},
        "piece_b_non_regression_classification_report": report_reference_classification or {},
        "piece_b_non_regression_localization_active_metrics": (report_reference_localization or {}).get("active_metrics_on_eval_set") or {},
        "piece_b_non_regression_localization_candidate_metrics": (report_reference_localization or {}).get("candidate_metrics_on_eval_set") or {},
        "piece_b_non_regression_classification_active_metrics": (report_reference_classification or {}).get("active_metrics_on_eval_set") or {},
        "piece_b_non_regression_classification_candidate_metrics": (report_reference_classification or {}).get("candidate_metrics_on_eval_set") or {},
        "piece_b_non_regression_localization_metric_delta": (report_reference_localization or {}).get("metric_delta"),
        "piece_b_non_regression_classification_metric_delta": (report_reference_classification or {}).get("metric_delta"),
        "localization_runtime_contract_path": localization_runtime_contract_path,
        "classification_runtime_contract_path": classification_runtime_contract_path,
        "active_false_negatives": active_fn,
        "candidate_false_negatives": candidate_fn,
        "fn_delta": fn_delta,
        "active_good_red_count": active_good_red_count,
        "candidate_good_red_count": candidate_good_red_count,
        "good_red_delta": good_red_delta,
        "operational_alerts": list(classification.get("operational_alerts") or []),
        "evaluation_started_at": started_at.isoformat(),
        "evaluation_completed_at": completed_at.isoformat(),
        "evaluation_duration_seconds": (completed_at - started_at).total_seconds(),
    }


def evaluate_reference_promotion_comparison(
    args: argparse.Namespace,
    *,
    cycle_dir: Path,
    reference_evaluation_set_path: Path,
    reference_evaluation_set_id: str,
    active_model_version: str,
    active_checkpoint_path: Path,
    active_localization_runtime: ActiveRuntimeModel | None = None,
    candidate_version: str,
    candidate_checkpoint_path: Path,
    localization_candidate_checkpoint_path: Path | None = None,
    classification_candidate_checkpoint_path: Path | None = None,
    artifacts: LifecycleArtifacts,
    state: CycleState,
    active_runtime: ActiveRuntimeModel,
) -> dict[str, Any]:
    started_at = datetime.now(UTC)
    if args.dual_promotion:
        return evaluate_dual_reference_promotion_comparison(
            args,
            cycle_dir=cycle_dir,
            reference_evaluation_set_path=reference_evaluation_set_path,
            reference_evaluation_set_id=reference_evaluation_set_id,
            active_model_version=active_model_version,
            active_checkpoint_path=active_checkpoint_path,
            active_localization_runtime=active_localization_runtime or active_runtime,
            candidate_version=candidate_version,
            localization_candidate_checkpoint_path=localization_candidate_checkpoint_path or candidate_checkpoint_path,
            classification_candidate_checkpoint_path=classification_candidate_checkpoint_path or candidate_checkpoint_path,
            artifacts=artifacts,
            state=state,
            active_runtime=active_runtime,
            started_at=started_at,
        )
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
    completed_at = datetime.now(UTC)

    reference_delta = reference.get("metric_delta")
    reference_false_negatives_ok = int(reference.get("candidate_false_negatives") or 0) <= int(
        reference.get("active_false_negatives") or 0
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
    active_image_recall = _finite_float(active_metrics.get("image_recall"))
    candidate_image_recall = _finite_float(candidate_metrics.get("image_recall"))
    image_recall_delta = (
        None
        if active_image_recall is None or candidate_image_recall is None
        else candidate_image_recall - active_image_recall
    )
    localization_gate = {
        "metric": reference.get("selected_metric"),
        "active_value": reference.get("active_metric_value"),
        "candidate_value": reference.get("candidate_metric_value"),
        "delta": reference.get("metric_delta"),
        "min_delta": float(args.promotion_min_delta),
        "thresholds_ok": thresholds_ok,
        "passed": bool(metric_ok and thresholds_ok),
    }
    classification_gate = {
        "active_false_negatives": reference.get("active_false_negatives"),
        "candidate_false_negatives": reference.get("candidate_false_negatives"),
        "fn_delta": int(reference.get("candidate_false_negatives") or 0)
        - int(reference.get("active_false_negatives") or 0),
        "active_image_recall": active_image_recall,
        "candidate_image_recall": candidate_image_recall,
        "image_recall_delta": image_recall_delta,
        "active_good_red_count": active_good_red_count,
        "candidate_good_red_count": candidate_good_red_count,
        "good_red_delta": good_red_delta,
        "active_good_red_rate": reference.get("active_good_red_rate"),
        "candidate_good_red_rate": reference.get("candidate_good_red_rate"),
        "max_good_red_regression": int(args.max_good_red_regression),
        "false_negatives_ok": reference_false_negatives_ok,
        "good_red_ok": good_red_ok,
        "passed": bool(reference_false_negatives_ok and good_red_ok),
    }
    classification_progress = _classification_progress_summary(
        classification_gate,
        reference_false_negatives_ok=reference_false_negatives_ok,
        good_red_ok=good_red_ok,
    )
    passed = bool(localization_gate["passed"] and classification_gate["passed"])

    mvp_gate = {
        "decision": "passed" if passed else "rejected",
        "selected_metric": reference.get("selected_metric"),
        "active_metric_value": reference.get("active_metric_value"),
        "candidate_metric_value": reference.get("candidate_metric_value"),
        "metric_delta": reference.get("metric_delta"),
        "metric_ok": metric_ok,
        "false_negatives_ok": reference_false_negatives_ok,
        "active_false_negatives": reference.get("active_false_negatives"),
        "candidate_false_negatives": reference.get("candidate_false_negatives"),
        "active_good_red_count": active_good_red_count,
        "candidate_good_red_count": candidate_good_red_count,
        "good_red_delta": good_red_delta,
        "max_good_red_regression": int(args.max_good_red_regression),
        "good_red_ok": good_red_ok,
        "thresholds_ok": thresholds_ok,
        "gate_eval_profile": args.gate_eval_profile,
        "localization_gate": localization_gate,
        "classification_gate": classification_gate,
        "classification_progress": classification_progress,
    }

    missing_comparable_metric = reference.get("selected_metric") is None
    if missing_comparable_metric:
        gate_reason = "rejected_missing_comparable_metric"
        promotion_status = "rejected_missing_comparable_metric"
        passed = False
    elif passed:
        gate_reason = "candidate_passed_representative_validation_gate"
        promotion_status = "promoted"
    elif not reference_false_negatives_ok:
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
    runtime_contract_path = write_feature_ae_runtime_contract(
        output_path=cycle_dir / "runtime_contracts" / "classification" / "runtime_contract.json",
        metrics_path=reference.get("candidate_eval_metrics_path"),
        checkpoint_path=candidate_checkpoint_path,
        model_version=candidate_version,
        model_role="classification",
        decision_thresholds=candidate_thresholds,
        scenario_id=args.scenario_id,
        lifecycle_run_id=str(getattr(args, "lifecycle_run_id", "")),
        cycle_id=cycle_dir.name,
    )

    return {
        "metrics": reference.get("metrics") or {},
        "active_metrics_on_eval_set": reference.get("active_metrics_on_eval_set") or {},
        "candidate_metrics_on_eval_set": reference.get("candidate_metrics_on_eval_set") or {},
        "reference_active_metrics_on_eval_set": reference.get("active_metrics_on_eval_set") or {},
        "reference_candidate_metrics_on_eval_set": reference.get("candidate_metrics_on_eval_set") or {},
        "reference_selected_metric": reference.get("selected_metric"),
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
        "gate_decision": "passed" if passed else "rejected",
        "gate_reason": gate_reason,
        "promotion_status": promotion_status,
        "promotion_panel_decision": mvp_gate,
        "simplified_gate": mvp_gate,
        "mvp_gate": mvp_gate,
        "localization_gate": localization_gate,
        "classification_gate": classification_gate,
        "classification_progress": classification_progress,
        "classification_progress_improved": classification_progress["improved"],
        "classification_progress_non_regression": classification_progress["non_regression"],
        "classification_progress_summary": classification_progress["summary"],
        "gate_eval_profile": args.gate_eval_profile,
        "active_decision_thresholds": active_thresholds,
        "active_eval_metrics_path": reference.get("active_eval_metrics_path"),
        "candidate_eval_metrics_path": reference.get("candidate_eval_metrics_path"),
        "reference_active_eval_metrics_path": reference.get("active_eval_metrics_path"),
        "reference_candidate_eval_metrics_path": reference.get("candidate_eval_metrics_path"),
        "active_cache_status": reference.get("active_cache_status"),
        "candidate_cache_status": reference.get("candidate_cache_status"),
        "active_cache_key": reference.get("active_cache_key"),
        "candidate_cache_key": reference.get("candidate_cache_key"),
        "cache_status": reference.get("cache_status"),
        "cache_key": reference.get("cache_key"),
        "cache_hit": reference.get("cache_hit"),
        "cache_source": reference.get("cache_source"),
        "candidate_decision_thresholds": candidate_thresholds,
        "runtime_contract_path": runtime_contract_path,
        "classification_runtime_contract_path": runtime_contract_path,
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
        "candidate_metric_timings": reference.get("candidate_metric_timings", {}),
        "active_metric_timings": reference.get("active_metric_timings", {}),
        "threshold_source": candidate_thresholds.get("threshold_source") if candidate_thresholds else "",
        "candidate_false_negatives": reference.get("candidate_false_negatives"),
        "active_false_negatives": reference.get("active_false_negatives"),
        "reference_candidate_false_negatives": reference.get("candidate_false_negatives"),
        "reference_active_false_negatives": reference.get("active_false_negatives"),
        "evaluation_started_at": started_at.isoformat(),
        "evaluation_completed_at": completed_at.isoformat(),
        "evaluation_duration_seconds": (completed_at - started_at).total_seconds(),
        "operational_alerts": operational_alerts,
    }


def evaluate_reference_model_on_set(
    args: argparse.Namespace,
    *,
    model_version: str,
    checkpoint_path: Path,
    evaluation_set_path: Path,
    output_dir: Path,
    evaluation_set_id: str,
    cache_root: Path,
    gt_masks_manifest: Path | None = None,
    cache_enabled: bool = False,
) -> dict[str, Any]:
    cache_key = metric_cache_key(
        model_version=model_version,
        checkpoint_path=checkpoint_path,
        evaluation_set_path=evaluation_set_path,
        evaluation_set_id=evaluation_set_id,
        gt_masks_manifest=gt_masks_manifest or args.reference_gt_masks_manifest,
        gate_eval_profile=args.gate_eval_profile,
        threshold_orange=0.02,
        threshold_red=0.05,
    )
    cached = load_metric_cache(cache_root, cache_key, output_dir) if cache_enabled else None
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
            threshold_orange=0.02,
            threshold_red=0.05,
            metric_profile=args.gate_eval_profile,
        )
    )
    params = json.loads((output_dir / "params.json").read_text(encoding="utf-8")) if (output_dir / "params.json").is_file() else {}
    metrics = {
        metric: float(value)
        for metric, value in (result.get("metrics") or {}).items()
        if value is not None and metric in GATE_METRICS_TO_KEEP
    }
    image_count = len(result.get("images") or [])
    payload = {
        "model_version": model_version,
        "checkpoint_path": str(checkpoint_path),
        "metrics": metrics,
        "metrics_path": str(output_dir / "metrics.json"),
        "params_path": str(output_dir / "params.json"),
        "metric_timings": result.get("metric_timings") or {},
        "eval_inference_seconds": params.get("duration_seconds"),
        "gate_eval_profile": args.gate_eval_profile,
        "image_count": image_count,
        "cache_status": "miss" if cache_enabled else "not_cached",
        "cache_hit": False,
        "cache_key": cache_key,
        "cache_source": str(cache_root / cache_key),
    }
    return payload


def metric_cache_key(
    *,
    model_version: str,
    checkpoint_path: Path,
    evaluation_set_path: Path,
    evaluation_set_id: str,
    gt_masks_manifest: Path | None,
    gate_eval_profile: str,
    threshold_orange: float,
    threshold_red: float,
) -> str:
    payload = {
        "model_version": model_version,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "evaluation_set_path": str(evaluation_set_path),
        "evaluation_set_id": evaluation_set_id,
        "evaluation_set_sha256": sha256_file(evaluation_set_path),
        "gt_masks_manifest_path": str(gt_masks_manifest) if gt_masks_manifest else "",
        "gt_masks_manifest_sha256": sha256_file(gt_masks_manifest) if gt_masks_manifest and gt_masks_manifest.is_file() else "",
        "score_contract_version": CANONICAL_FEATURE_AE_PREPROCESSING.version,
        "score_image": CANONICAL_FEATURE_AE_PREPROCESSING.score_image,
        "topk_fraction": CANONICAL_FEATURE_AE_PREPROCESSING.topk_fraction,
        "roi_threshold": CANONICAL_FEATURE_AE_PREPROCESSING.roi_threshold,
        "threshold_orange": float(threshold_orange),
        "threshold_red": float(threshold_red),
        "gate_eval_profile": gate_eval_profile,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_metric_cache(cache_root: Path, cache_key: str, output_dir: Path) -> dict[str, Any] | None:
    cache_dir = cache_root / cache_key
    metrics_path = cache_dir / "metrics.json"
    if not metrics_path.is_file():
        return None
    params_path = cache_dir / "params.json"
    if not params_path.is_file():
        return None
    params: dict[str, Any] = json.loads(params_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("metrics.json", "params.json"):
        source = cache_dir / name
        if source.is_file():
            shutil.copy2(source, output_dir / name)
    payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics = {
        metric: float(value)
        for metric, value in (payload.get("metrics") or {}).items()
        if value is not None and metric in GATE_METRICS_TO_KEEP
    }
    image_count = len(payload.get("images") or [])
    return {
        "metrics": metrics,
        "metrics_path": str(output_dir / "metrics.json"),
        "params_path": str(output_dir / "params.json"),
        "metric_timings": payload.get("metric_timings") or {},
        "eval_inference_seconds": params.get("duration_seconds"),
        "gate_eval_profile": params.get("metric_profile"),
        "image_count": image_count,
    }


def store_metric_cache(cache_root: Path, cache_key: str, output_dir: Path) -> None:
    cache_dir = cache_root / cache_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name in ("metrics.json", "params.json"):
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
    *,
    priority: tuple[str, ...] | None = None,
) -> str | None:
    for metric in priority or FEATURE_AE_BUSINESS_METRIC_PRIORITY:
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


def write_feature_ae_runtime_contract(
    *,
    output_path: Path,
    metrics_path: str | Path | None,
    checkpoint_path: Path,
    model_version: str,
    model_role: str,
    decision_thresholds: dict[str, Any] | None,
    scenario_id: str,
    lifecycle_run_id: str,
    cycle_id: str,
) -> str:
    if not metrics_path:
        return ""
    metrics_file = Path(str(metrics_path))
    if not metrics_file.is_file():
        return ""
    payload = json.loads(metrics_file.read_text(encoding="utf-8"))
    score_contract = payload.get("score_contract")
    if not isinstance(score_contract, dict):
        return ""
    thresholds = decision_thresholds or payload.get("decision_thresholds")
    if not isinstance(thresholds, dict):
        return ""
    reference_contract = feature_ae_reference_contract_from_payload(score_contract).to_dict()
    contract = {
        "runtime_contract_version": "feature_ae_runtime_contract_v001",
        "model_type": "feature_ae_reference",
        "model_version": model_version,
        "model_role": model_role,
        "scenario_id": scenario_id,
        "lifecycle_run_id": lifecycle_run_id,
        "cycle_id": cycle_id,
        "checkpoint_filename": checkpoint_path.name,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "feature_ae_reference_contract": reference_contract,
        "source_score_contract": score_contract,
        "decision_thresholds": thresholds,
        "validation_set_id": payload.get("validation_set_id"),
        "source_metrics_path": safe_checkpoint_reference(metrics_file),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(output_path)


def reference_contract_from_runtime_contract_path(path: str | Path | None) -> Any | None:
    if not path:
        return None
    runtime_contract_path = Path(str(path))
    if not runtime_contract_path.is_file():
        return None
    runtime_contract = load_feature_ae_runtime_contract(runtime_contract_path)
    return feature_ae_reference_contract_from_payload(
        runtime_contract.get("feature_ae_reference_contract") or runtime_contract["score_contract"]
    )


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


def select_classification_candidate_checkpoint(
    args: argparse.Namespace,
    *,
    cycle_dir: Path,
    candidate_version: str,
    default_checkpoint: Path,
    active_runtime: ActiveRuntimeModel,
    artifacts: LifecycleArtifacts,
    state: CycleState,
) -> dict[str, Any]:
    manifest_value = getattr(args, "classification_selection_manifest", None)
    manifest_path = Path(str(manifest_value)) if manifest_value else None
    if manifest_path is None or not manifest_path.is_file():
        return {
            "enabled": False,
            "reason": "classification_selection_manifest_missing",
            "selected_checkpoint": str(default_checkpoint),
        }

    candidate_checkpoints = _classification_selection_checkpoint_candidates(
        progressive_candidate_run_dir(args, candidate_version),
        default_checkpoint=default_checkpoint,
    )
    if not candidate_checkpoints:
        return {
            "enabled": True,
            "reason": "no_candidate_checkpoints",
            "selection_manifest": str(manifest_path),
            "selected_checkpoint": str(default_checkpoint),
        }

    evaluated: list[dict[str, Any]] = []
    active_good_red_count: int | None = None
    for checkpoint in candidate_checkpoints:
        panel_name = f"classification_selection_{checkpoint.stem}"
        evaluation = evaluate_model_pair_on_panel(
            args,
            cycle_dir=cycle_dir,
            panel_name=panel_name,
            evaluation_set_path=manifest_path,
            evaluation_set_id=manifest_path.stem,
            active_model_version=active_runtime.version,
            active_checkpoint_path=active_runtime.checkpoint,
            candidate_version=candidate_version,
            candidate_checkpoint_path=checkpoint,
            artifacts=artifacts,
            state=state,
            active_runtime=active_runtime,
            active_decision_thresholds=active_runtime.decision_thresholds,
            gt_masks_manifest=args.reference_gt_masks_manifest,
            metric_priority=CLASSIFICATION_METRIC_PRIORITY,
        )
        active_metrics = evaluation.get("active_metrics_on_eval_set") or {}
        candidate_metrics = evaluation.get("candidate_metrics_on_eval_set") or {}
        if active_good_red_count is None:
            active_good_red_count = int(active_metrics.get("good_red_count") or 0)
        candidate_good_red_count = int(candidate_metrics.get("good_red_count") or 0)
        candidate_fn = int(candidate_metrics.get("false_negatives") or 0)
        candidate_recall = _finite_float(candidate_metrics.get("image_recall"))
        candidate_ap = _finite_float(candidate_metrics.get("image_ap"))
        good_red_delta = candidate_good_red_count - int(active_good_red_count or 0)
        good_red_ok = good_red_delta <= int(args.max_good_red_regression)
        evaluated.append(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "false_negatives": candidate_fn,
                "image_recall": candidate_recall,
                "image_ap": candidate_ap,
                "good_red_count": candidate_good_red_count,
                "active_good_red_count": int(active_good_red_count or 0),
                "good_red_delta": good_red_delta,
                "good_red_ok": good_red_ok,
                "metrics_path": evaluation.get("candidate_eval_metrics_path"),
            }
        )

    selectable = [row for row in evaluated if row["good_red_ok"]]
    if not selectable:
        selected = next((row for row in evaluated if row["checkpoint"] == str(default_checkpoint)), evaluated[0])
        return {
            "enabled": True,
            "reason": "no_checkpoint_passed_good_red_guard",
            "selection_manifest": str(manifest_path),
            "selected_checkpoint": selected["checkpoint"],
            "selected_checkpoint_sha256": selected["checkpoint_sha256"],
            "candidates_evaluated": evaluated,
        }

    selected = max(
        selectable,
        key=lambda row: (
            -int(row["false_negatives"]),
            float(row["image_recall"]) if row["image_recall"] is not None else float("-inf"),
            float(row["image_ap"]) if row["image_ap"] is not None else float("-inf"),
        ),
    )
    return {
        "enabled": True,
        "reason": "selected_by_false_negatives_recall_ap",
        "selection_manifest": str(manifest_path),
        "selected_checkpoint": selected["checkpoint"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "selected_false_negatives": selected["false_negatives"],
        "selected_image_recall": selected["image_recall"],
        "selected_image_ap": selected["image_ap"],
        "selected_good_red_count": selected["good_red_count"],
        "active_good_red_count": selected["active_good_red_count"],
        "selected_good_red_delta": selected["good_red_delta"],
        "candidates_evaluated": evaluated,
    }


def _classification_selection_checkpoint_candidates(run_dir: Path, *, default_checkpoint: Path) -> list[Path]:
    candidates = [
        *sorted(run_dir.glob("checkpoint_epoch_*.pt")),
        run_dir / "checkpoint_best_image.pt",
        run_dir / "checkpoint_best_image_ap.pt",
        run_dir / "checkpoint_best_image_auroc.pt",
        default_checkpoint,
    ]
    output: list[Path] = []
    seen_hashes: set[str] = set()
    for candidate in candidates:
        if not candidate.is_file():
            continue
        digest = sha256_file(candidate)
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        output.append(candidate)
    return output


def resolve_specialized_candidate_checkpoints(train_result: dict[str, Any]) -> dict[str, Path]:
    default_checkpoint = Path(str(train_result.get("checkpoint") or ""))
    run_dir = Path(str(train_result.get("run_dir") or default_checkpoint.parent))
    localization = _first_existing_or_default(
        [
            run_dir / "checkpoint_best_localization.pt",
            run_dir / "checkpoint_best_pixel_aupimo_1e-5_1e-3.pt",
            run_dir / "checkpoint_best_pixel_ap.pt",
        ],
        default_checkpoint,
    )
    classification = _first_existing_or_default(
        [
            run_dir / "checkpoint_best_image.pt",
            run_dir / "checkpoint_best_image_ap.pt",
            run_dir / "checkpoint_best_image_auroc.pt",
        ],
        default_checkpoint,
    )
    return {"localization": localization, "classification": classification}


def validate_cycle_checkpoints(
    *,
    candidate_checkpoint: Path,
    localization_checkpoint: Path,
    classification_checkpoint: Path,
) -> dict[str, Any]:
    candidate = validate_feature_ae_checkpoint_file(candidate_checkpoint, role="candidate")
    localization = validate_feature_ae_checkpoint_file(localization_checkpoint, role="localization")
    classification = validate_feature_ae_checkpoint_file(classification_checkpoint, role="classification")
    return {
        "candidate_checkpoint_sha256": candidate["sha256"],
        "candidate_checkpoint_size_bytes": candidate["size_bytes"],
        "localization_checkpoint_sha256": localization["sha256"],
        "localization_checkpoint_size_bytes": localization["size_bytes"],
        "classification_checkpoint_sha256": classification["sha256"],
        "classification_checkpoint_size_bytes": classification["size_bytes"],
    }


def validate_feature_ae_checkpoint_file(path: Path, *, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{role} checkpoint is missing: {path}")
    size_bytes = path.stat().st_size
    if size_bytes < MIN_VALID_CHECKPOINT_BYTES:
        raise ValueError(
            f"{role} checkpoint is too small to be a valid Feature-AE checkpoint: "
            f"{path} ({size_bytes} bytes)"
        )
    try:
        import torch

        torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(f"{role} checkpoint cannot be loaded: {path}: {type(exc).__name__}: {exc}") from exc
    return {"path": str(path), "size_bytes": size_bytes, "sha256": sha256_file(path)}


def _first_existing_or_default(candidates: list[Path], default: Path) -> Path:
    return next((path for path in candidates if path.is_file()), default)


def _dual_rejection_status(gate: dict[str, Any], *, role: str) -> str:
    if role == "localization":
        if gate.get("metric") is None:
            return "rejected_missing_comparable_metric"
        return "rejected_reference_regression"
    if not gate.get("thresholds_ok", True):
        return "rejected_missing_runtime_thresholds"
    if not gate.get("good_red_ok", True):
        return "rejected_operational_guardrail"
    return "rejected_no_classification_improvement"


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
    *,
    model_name_base: str = "feature_ae",
    model_role: str = "",
) -> dict[str, Any]:
    run_id = str(cycle.get("mlflow_run_id") or "")
    if not run_id:
        return {"registry_status": "skipped", "registry_reason": "missing_mlflow_run_id"}
    try:
        model_artifact_path = {
            "localization": "model_localization",
            "classification": "model_classification",
        }.get(model_role, "model")
        result = register_run_to_model(
            run_id=run_id,
            scenario_id=state.scenario_id,
            stage=args.target_stage,
            model_name_base=model_name_base,
            model_artifact_path=model_artifact_path,
        )
    except Exception as exc:
        if args.require_mlflow_registry:
            raise
        status = "failed_missing_mlflow_model" if "missing_mlflow_model_artifact" in str(exc) else "failed"
        return {
            "registry_status": status,
            "registry_reason": str(exc),
            "registered_model_name": registered_model_name(state.scenario_id, base_name=model_name_base),
            "registry_stage": args.target_stage,
            "registry_source_of_truth": "mlflow_registry",
            "model_role": model_role,
        }
    return {
        "registry_status": "registered",
        "registered_model_name": result.get("registered_model_name"),
        "registered_model_version": result.get("version"),
        "registry_alias": result.get("alias") or result.get("stage") or args.target_stage,
        "registry_stage": result.get("stage") or args.target_stage,
        "registry_source_of_truth": result.get("source_of_truth") or "mlflow_registry",
        "model_artifact_path": result.get("model_artifact_path") or {
            "localization": "model_localization",
            "classification": "model_classification",
        }.get(model_role, "model"),
        "model_role": model_role,
    }


def prefix_registry_result(prefix: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_registry_status": result.get("registry_status"),
        f"{prefix}_registry_reason": result.get("registry_reason"),
        f"{prefix}_registered_model_name": result.get("registered_model_name"),
        f"{prefix}_registered_model_version": result.get("registered_model_version"),
        f"{prefix}_registry_alias": result.get("registry_alias"),
        f"{prefix}_registry_stage": result.get("registry_stage"),
        f"{prefix}_registry_source_of_truth": result.get("registry_source_of_truth"),
        f"{prefix}_model_artifact_path": result.get("model_artifact_path"),
    }


def role_promotion_cycle(cycle: dict[str, Any], role: str) -> dict[str, Any]:
    role_gate = cycle.get(f"{role}_gate") if role in {"localization", "classification"} else None
    payload = dict(cycle)
    payload["model_role"] = role
    payload["promotion_status"] = cycle.get(f"{role}_promotion_status")
    payload["registered_model_name"] = cycle.get(f"{role}_registered_model_name")
    payload["registered_model_version"] = cycle.get(f"{role}_registered_model_version")
    payload["registry_alias"] = cycle.get(f"{role}_registry_alias")
    payload["registry_stage"] = cycle.get(f"{role}_registry_stage")
    payload["registry_status"] = cycle.get(f"{role}_registry_status")
    if role == "localization":
        payload["selected_metric"] = cycle.get("localization_selected_metric")
        payload["active_metric_value"] = cycle.get("localization_active_metric_value")
        payload["candidate_metric_value"] = cycle.get("localization_candidate_metric_value")
        payload["metric_delta"] = cycle.get("localization_metric_delta")
        payload["candidate_checkpoint"] = cycle.get("localization_candidate_checkpoint")
        payload["checkpoint_filename"] = Path(str(payload["candidate_checkpoint"] or "")).name
        payload["checkpoint_sha256"] = cycle.get("localization_checkpoint_sha256")
        payload["model_artifact_path"] = cycle.get("localization_model_artifact_path")
        payload["runtime_contract_path"] = cycle.get("localization_runtime_contract_path")
    elif role == "classification":
        payload["selected_metric"] = cycle.get("classification_selected_metric")
        payload["active_metric_value"] = cycle.get("classification_active_metric_value")
        payload["candidate_metric_value"] = cycle.get("classification_candidate_metric_value")
        payload["metric_delta"] = cycle.get("classification_metric_delta")
        payload["candidate_checkpoint"] = cycle.get("classification_candidate_checkpoint")
        payload["checkpoint_filename"] = Path(str(payload["candidate_checkpoint"] or "")).name
        payload["checkpoint_sha256"] = cycle.get("classification_checkpoint_sha256")
        payload["model_artifact_path"] = cycle.get("classification_model_artifact_path")
        payload["runtime_contract_path"] = cycle.get("classification_runtime_contract_path") or cycle.get("runtime_contract_path")
    payload["lifecycle_run_id"] = cycle.get("run_id") or cycle.get("lifecycle_run_id")
    payload["candidate_run_dir"] = cycle.get("candidate_run_dir")
    payload["source_mlflow_run_id"] = cycle.get("mlflow_run_id")
    if isinstance(role_gate, dict):
        payload["gate_decision"] = "passed" if role_gate.get("passed") else "rejected"
        payload["gate_reason"] = f"{role}_{payload['promotion_status']}"
    return payload


def publish_promoted_checkpoints(args: argparse.Namespace, state: CycleState, cycle: dict[str, Any]) -> None:
    if not args.publish_minio:
        return
    cycle_id = str(cycle.get("cycle_id") or "")
    if not cycle_id:
        return
    role_specs = []
    if cycle.get("localization_promotion_status") == "promoted":
        role_specs.append(("localization", cycle.get("localization_candidate_checkpoint")))
    if cycle.get("classification_promotion_status") == "promoted":
        role_specs.append(("classification", cycle.get("classification_candidate_checkpoint")))
    if not role_specs and cycle.get("promotion_status") == "promoted":
        role_specs.append(("classification", cycle.get("candidate_checkpoint")))
    for role, checkpoint in role_specs:
        if not checkpoint:
            continue
        uri = f"s3://iqa-models/{state.scenario_id}/{state.run_id}/{cycle_id}/{role}/checkpoint.pt"
        upload_checkpoint_to_s3(str(checkpoint), uri)
        cycle[f"{role}_checkpoint_s3_uri"] = uri


def tag_mlflow_promotion_evidence(cycle: dict[str, Any]) -> None:
    run_id = str(cycle.get("mlflow_run_id") or "")
    if not run_id:
        cycle["mlflow_gate_evidence_status"] = "skipped_missing_run_id"
        return
    cycle["mlflow_gate_evidence_status"] = "logging"
    try:
        import mlflow

        _configure_mlflow_tracking(mlflow)
        client = mlflow.tracking.MlflowClient()
        tag_keys = [
            "model_role",
            "scenario_id",
            "lifecycle_run_id",
            "cycle_id",
            "promotion_policy",
            "promotion_objective",
            "piece_b_non_regression_policy",
            "active_model_before",
            "classification_model_before",
            "localization_model_before",
            "candidate_version",
            "candidate_run_dir",
            "checkpoint_filename",
            "checkpoint_sha256",
            "runtime_contract_path",
            "source_mlflow_run_id",
            "model_artifact_path",
            "evaluation_set_id",
            "gate_evaluation_set_id",
            "gate_evaluation_set_path",
            "reference_evaluation_report_set_id",
            "reference_evaluation_report_set_path",
            "evaluation_seen_events",
            "gate_eval_profile",
            "candidate_init_policy",
            "candidate_initial_model_version",
            "candidate_initial_checkpoint",
            "candidate_initial_checkpoint_sha256",
            "classification_selection_manifest",
            "classification_selection_reason",
            "classification_selection_checkpoint_sha256",
            "classification_selection_false_negatives",
            "classification_selection_image_recall",
            "classification_selection_image_ap",
            "selected_metric",
            "active_metric_value",
            "candidate_metric_value",
            "metric_delta",
            "gate_decision",
            "gate_reason",
            "promotion_status",
            "registered_model_name",
            "registry_stage",
            "registry_status",
            "activated_for_next_events",
            "activation_scope",
            "threshold_source",
            "active_false_negatives",
            "candidate_false_negatives",
            "active_good_red_count",
            "candidate_good_red_count",
            "good_red_delta",
            "fn_delta",
            "registered_model_version",
            "registry_alias",
            "active_cache_status",
            "candidate_cache_status",
            "active_cache_key",
            "candidate_cache_key",
            "classification_progress_improved",
            "classification_progress_non_regression",
            "classification_progress_summary",
            "dual_promotion",
            "localization_promotion_status",
            "classification_promotion_status",
            "localization_gate_reason",
            "classification_gate_reason",
            "localization_registered_model_name",
            "classification_registered_model_name",
            "localization_registered_model_version",
            "classification_registered_model_version",
            "localization_registry_status",
            "classification_registry_status",
            "localization_registry_alias",
            "classification_registry_alias",
            "localization_checkpoint_sha256",
            "classification_checkpoint_sha256",
            "candidate_checkpoint_sha256",
        ]
        version_name = str(cycle.get("registered_model_name") or "")
        version_number = str(cycle.get("registered_model_version") or "")
        for key in tag_keys:
            value = cycle.get(key)
            if value is not None:
                tag_value = str(value)
                client.set_tag(run_id, key, tag_value)
                if version_name and version_number:
                    client.set_model_version_tag(version_name, version_number, key, tag_value)
        for key, value in _mlflow_gate_params(cycle).items():
            if value is not None:
                try:
                    client.log_param(run_id, key, str(value))
                except Exception:
                    client.set_tag(run_id, key, str(value))
        for key, value in _mlflow_gate_metrics(cycle).items():
            metric_value = _finite_float(value)
            if metric_value is not None:
                client.log_metric(run_id, key, metric_value)
        _log_mlflow_gate_artifacts(client, run_id, cycle)
        if version_name and version_number:
            description = (
                f"{cycle.get('candidate_version', version_name)}: "
                f"gate={cycle.get('gate_decision')} "
                f"promotion={cycle.get('promotion_status')} "
                f"metric={cycle.get('selected_metric')} "
                f"delta={cycle.get('metric_delta')} "
                f"fn={cycle.get('candidate_false_negatives')} "
                f"good_red={cycle.get('candidate_good_red_count')}"
            )
            client.update_model_version(version_name, version_number, description=description)
        cycle["mlflow_gate_evidence_status"] = "logged"
        client.set_tag(run_id, "mlflow_gate_evidence_status", "logged")
        if version_name and version_number:
            client.set_model_version_tag(version_name, version_number, "mlflow_gate_evidence_status", "logged")
    except Exception as exc:
        cycle["mlflow_gate_evidence_status"] = "failed"
        cycle["mlflow_gate_evidence_error"] = f"{type(exc).__name__}: {exc}"
        return


def _configure_mlflow_tracking(mlflow_module: Any) -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or os.getenv("IQA_MLFLOW_TRACKING_URI")
    if tracking_uri:
        os.environ.setdefault("MLFLOW_TRACKING_URI", tracking_uri)
        mlflow_module.set_tracking_uri(tracking_uri)
    s3_endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL") or os.getenv("IQA_S3_ENDPOINT_URL")
    if s3_endpoint:
        os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", s3_endpoint)


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _classification_progress_summary(
    classification_gate: dict[str, Any],
    *,
    reference_false_negatives_ok: bool,
    good_red_ok: bool,
) -> dict[str, Any]:
    fn_delta = int(classification_gate.get("fn_delta") or 0)
    recall_delta = _finite_float(classification_gate.get("image_recall_delta"))
    improved = bool(fn_delta < 0 or (recall_delta is not None and recall_delta > 0))
    non_regression = bool(reference_false_negatives_ok and good_red_ok)
    parts = [
        f"FN {classification_gate.get('active_false_negatives')} -> {classification_gate.get('candidate_false_negatives')}",
        f"good red {classification_gate.get('active_good_red_count')} -> {classification_gate.get('candidate_good_red_count')}",
    ]
    if recall_delta is not None:
        parts.append(
            "recall "
            f"{float(classification_gate.get('active_image_recall') or 0.0):.3f} -> "
            f"{float(classification_gate.get('candidate_image_recall') or 0.0):.3f}"
        )
    status = "improved" if improved else "stable"
    if not non_regression:
        status = "regressed"
    return {
        "improved": improved,
        "non_regression": non_regression,
        "summary": f"{status}: " + ", ".join(parts),
    }


def _mlflow_gate_params(cycle: dict[str, Any]) -> dict[str, Any]:
    return {
        "gate.decision": cycle.get("gate_decision"),
        "gate.reason": cycle.get("gate_reason"),
        "gate.promotion_status": cycle.get("promotion_status"),
        "gate.promotion_objective": cycle.get("promotion_objective"),
        "gate.piece_b_non_regression_policy": cycle.get("piece_b_non_regression_policy"),
        "gate.eval_profile": cycle.get("gate_eval_profile"),
        "gate.selected_metric": cycle.get("selected_metric"),
        "gate.evaluation_set_id": cycle.get("evaluation_set_id"),
        "gate.gate_evaluation_set_id": cycle.get("gate_evaluation_set_id"),
        "gate.reference_evaluation_report_set_id": cycle.get("reference_evaluation_report_set_id"),
        "gate.threshold_source": cycle.get("threshold_source"),
        "gate.active_cache_status": cycle.get("active_cache_status"),
        "gate.candidate_cache_status": cycle.get("candidate_cache_status"),
        "gate.dual_promotion": cycle.get("dual_promotion"),
        "gate.dual_promotion_outcome": cycle.get("dual_promotion_outcome"),
        "gate.localization.promotion_status": cycle.get("localization_promotion_status"),
        "gate.classification.promotion_status": cycle.get("classification_promotion_status"),
        "gate.localization.reason": cycle.get("localization_gate_reason"),
        "gate.classification.reason": cycle.get("classification_gate_reason"),
        "gate.localization.selected_metric": cycle.get("localization_selected_metric"),
        "gate.classification.selected_metric": cycle.get("classification_selected_metric"),
        "gate.localization.registry_status": cycle.get("localization_registry_status"),
        "gate.classification.registry_status": cycle.get("classification_registry_status"),
        "classification_progress.improved": cycle.get("classification_progress_improved"),
        "classification_progress.summary": cycle.get("classification_progress_summary"),
    }


def _mlflow_gate_metrics(cycle: dict[str, Any]) -> dict[str, Any]:
    gate_decision = str(cycle.get("gate_decision") or "")
    localization_promoted = cycle.get("localization_promotion_status") == "promoted"
    classification_promoted = cycle.get("classification_promotion_status") == "promoted"
    metrics: dict[str, Any] = {
        "gate.passed": gate_decision in {"passed", "partially_passed"},
        "gate.all_roles_promoted": localization_promoted and classification_promoted,
        "gate.any_role_promoted": localization_promoted or classification_promoted,
        "gate.active_metric_value": cycle.get("active_metric_value"),
        "gate.candidate_metric_value": cycle.get("candidate_metric_value"),
        "gate.metric_delta": cycle.get("metric_delta"),
        "gate.active_false_negatives": cycle.get("active_false_negatives"),
        "gate.candidate_false_negatives": cycle.get("candidate_false_negatives"),
        "gate.fn_delta": cycle.get("fn_delta"),
        "gate.active_good_red_count": cycle.get("active_good_red_count"),
        "gate.candidate_good_red_count": cycle.get("candidate_good_red_count"),
        "gate.good_red_delta": cycle.get("good_red_delta"),
        "gate.max_good_red_regression": cycle.get("max_good_red_regression"),
        "gate.localization.active_metric_value": cycle.get("localization_active_metric_value"),
        "gate.localization.candidate_metric_value": cycle.get("localization_candidate_metric_value"),
        "gate.localization.metric_delta": cycle.get("localization_metric_delta"),
        "gate.classification.active_metric_value": cycle.get("classification_active_metric_value"),
        "gate.classification.candidate_metric_value": cycle.get("classification_candidate_metric_value"),
        "gate.classification.metric_delta": cycle.get("classification_metric_delta"),
        "gate.classification.active_false_negatives": cycle.get("active_false_negatives"),
        "gate.classification.candidate_false_negatives": cycle.get("candidate_false_negatives"),
        "gate.piece_b_non_regression.localization.metric_delta": cycle.get("piece_b_non_regression_localization_metric_delta"),
        "gate.piece_b_non_regression.classification.metric_delta": cycle.get("piece_b_non_regression_classification_metric_delta"),
    }
    gate = cycle.get("mvp_gate") if isinstance(cycle.get("mvp_gate"), dict) else cycle.get("simplified_gate")
    if isinstance(gate, dict):
        for key in ("metric_ok", "false_negatives_ok", "good_red_ok", "thresholds_ok"):
            if key in gate:
                metrics[f"gate.{key}"] = gate[key]
    localization_gate = cycle.get("localization_gate")
    if isinstance(localization_gate, dict):
        for key in ("active_value", "candidate_value", "delta", "min_delta", "thresholds_ok", "passed"):
            if key in localization_gate:
                metrics[f"gate.localization.{key}"] = localization_gate[key]
    classification_gate = cycle.get("classification_gate")
    if isinstance(classification_gate, dict):
        for key in (
            "active_false_negatives",
            "candidate_false_negatives",
            "fn_delta",
            "active_image_recall",
            "candidate_image_recall",
            "image_recall_delta",
            "active_good_red_count",
            "candidate_good_red_count",
            "good_red_delta",
            "active_good_red_rate",
            "candidate_good_red_rate",
            "max_good_red_regression",
            "false_negatives_ok",
            "fn_non_regression",
            "fn_improved",
            "metric_ok",
            "good_red_ok",
            "passed",
        ):
            if key in classification_gate:
                metrics[f"gate.classification.{key}"] = classification_gate[key]
    classification_progress = cycle.get("classification_progress")
    if isinstance(classification_progress, dict):
        for key in ("improved", "non_regression"):
            if key in classification_progress:
                metrics[f"gate.classification_progress.{key}"] = classification_progress[key]
    for role, source_key in (
        ("active", "active_metrics_on_eval_set"),
        ("candidate", "candidate_metrics_on_eval_set"),
    ):
        source = cycle.get(source_key)
        if not isinstance(source, dict):
            continue
        for metric_name in (
            "pixel_aupimo_1e-5_1e-3",
            "image_ap",
            "image_auroc",
            "image_recall",
            "false_negatives",
            "good_red_count",
            "good_red_rate",
            "red_count",
            "red_rate",
            "alert_count",
            "alert_rate",
        ):
            if metric_name in source:
                metrics[f"gate.{role}.{metric_name}"] = source[metric_name]
    return metrics


def _log_mlflow_gate_artifacts(client: Any, run_id: str, cycle: dict[str, Any]) -> None:
    evidence_keys = [
        "cycle_id",
        "gate_decision",
        "gate_reason",
        "promotion_status",
        "promotion_policy",
        "gate_eval_profile",
        "selected_metric",
        "active_metric_value",
        "candidate_metric_value",
        "metric_delta",
        "active_false_negatives",
        "candidate_false_negatives",
        "fn_delta",
        "active_good_red_count",
        "candidate_good_red_count",
        "good_red_delta",
        "active_cache_status",
        "candidate_cache_status",
        "active_cache_key",
        "candidate_cache_key",
        "threshold_source",
        "mvp_gate",
        "simplified_gate",
        "localization_gate",
        "classification_gate",
        "classification_progress",
        "dual_promotion",
        "localization_promotion_status",
        "classification_promotion_status",
        "localization_candidate_checkpoint",
        "classification_candidate_checkpoint",
        "candidate_run_dir",
        "candidate_checkpoint_sha256",
        "localization_checkpoint_sha256",
        "classification_checkpoint_sha256",
        "localization_checkpoint_s3_uri",
        "classification_checkpoint_s3_uri",
        "localization_model_artifact_path",
        "classification_model_artifact_path",
        "localization_selected_metric",
        "classification_selected_metric",
        "localization_metric_delta",
        "classification_metric_delta",
        "localization_registered_model_name",
        "classification_registered_model_name",
        "localization_registered_model_version",
        "classification_registered_model_version",
        "localization_registry_status",
        "classification_registry_status",
        "localization_runtime_contract_path",
        "classification_runtime_contract_path",
        "runtime_contract_path",
    ]
    evidence = {key: cycle.get(key) for key in evidence_keys if key in cycle}
    with tempfile.TemporaryDirectory(prefix="iqa_gate_evidence_") as tmp:
        evidence_path = Path(tmp) / "gate_evidence.json"
        evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        client.log_artifact(run_id, str(evidence_path), artifact_path="gate")
    for key, artifact_path in (
        ("active_eval_metrics_path", "gate/active"),
        ("candidate_eval_metrics_path", "gate/candidate"),
    ):
        value = cycle.get(key)
        if not value:
            continue
        path = Path(str(value))
        if path.is_file():
            client.log_artifact(run_id, str(path), artifact_path=artifact_path)
    for role in ("localization", "classification"):
        value = cycle.get(f"{role}_runtime_contract_path")
        if not value and str(cycle.get("model_role") or "") == role:
            value = cycle.get("runtime_contract_path")
        if not value and role == "classification":
            value = cycle.get("runtime_contract_path")
        if not value:
            continue
        path = Path(str(value))
        if path.is_file():
            client.log_artifact(run_id, str(path), artifact_path=f"runtime_contracts/{role}")


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


def _best_role_cycle_id(cycles: list[dict[str, Any]], *, role: str) -> str | None:
    status_key = f"{role}_promotion_status"
    metric_key = f"{role}_metric_delta"
    candidates = [cycle for cycle in cycles if cycle.get(status_key) == "promoted"]
    if not candidates:
        return None
    best = max(candidates, key=lambda cycle: float(cycle.get(metric_key) or float("-inf")))
    return str(best.get("cycle_id")) if best else None


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
    if args.candidate_init_policy == "fresh":
        return None
    if args.candidate_init_policy == "active":
        return active_runtime.checkpoint
    return resolve_feature_ae_checkpoint(DEFAULT_FEATURE_AE_MODEL_VERSION, strict_checksum=True)


def resolve_candidate_initial_model_version(args: argparse.Namespace, *, active_runtime: ActiveRuntimeModel) -> str:
    if args.candidate_init_policy == "fresh":
        return "none"
    if args.candidate_init_policy == "active":
        return active_runtime.version
    return DEFAULT_FEATURE_AE_MODEL_VERSION


def safe_checkpoint_reference(path: Path | None) -> str:
    if path is None:
        return ""
    path = Path(path)
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        if not path.is_absolute():
            return path.as_posix()
        return path.name


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
    initial_checkpoint_reference: str = "",
    initial_checkpoint_sha256: str = "",
    initial_model_version: str = "",
    cycle_id: str = "",
) -> dict[str, Any]:
    model_cache_root = Path(getattr(args, "model_cache_root", DEFAULT_MODEL_CACHE_ROOT))
    run_dir = progressive_candidate_run_dir(args, candidate_version)
    _reset_generated_progressive_candidate_run_dir(run_dir, model_cache_root=model_cache_root)
    run_dir.mkdir(parents=True, exist_ok=True)

    def emit_epoch_metrics(epoch_record: dict[str, Any]) -> None:
        epoch_metrics = epoch_record.get("metrics") or {}
        emit_lifecycle_api_event(
            args,
            "epoch_completed",
            cycle_id=cycle_id,
            epoch=int(epoch_record.get("epoch") or 0),
            candidate_version=candidate_version,
            candidate_initial_model_version=initial_model_version,
            candidate_initial_checkpoint_sha256=initial_checkpoint_sha256,
            metrics={
                "pixel_aupimo_1e-5_1e-3": epoch_metrics.get("pixel_aupimo_1e-5_1e-3"),
                "pixel_ap": epoch_metrics.get("pixel_ap"),
                "image_ap": epoch_metrics.get("image_ap"),
            },
        )

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
        candidate_initial_model_version=initial_model_version,
        candidate_initial_checkpoint=initial_checkpoint_reference,
        candidate_initial_checkpoint_sha256=initial_checkpoint_sha256,
        lifecycle_run_id=str(getattr(args, "lifecycle_run_id", "") or ""),
        cycle_id=cycle_id,
        lifecycle_event_callback=emit_epoch_metrics,
        run_name=f"{candidate_version}_{args.target_stage}",
        device=args.device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_steps=args.max_steps,
        metric_eval_manifest_path=args.reference_eval_manifest,
        gt_masks_manifest=args.reference_gt_masks_manifest,
        metric_eval_device=args.device,
        metric_eval_every_epochs=1,
        metric_eval_start_epoch=1,
        metric_eval_calibrate_normal=False,
        metric_eval_layer_weights={"layer2": 0.65, "layer3": 0.35},
        metric_eval_apply_score_region_to_map=True,
        metric_eval_profile=args.gate_eval_profile,
        require_business_metric_for_early_stopping=True,
    )
    return train_feature_ae_with_mlflow_logging(config, git_commit=_git_commit())


def _reset_generated_progressive_candidate_run_dir(
    run_dir: Path,
    *,
    model_cache_root: Path = DEFAULT_MODEL_CACHE_ROOT,
) -> None:
    """Clear stale artifacts for deterministic progressive candidate versions."""
    if not run_dir.exists():
        return
    resolved_run_dir = run_dir.resolve()
    resolved_root = model_cache_root.resolve()
    if resolved_root not in (resolved_run_dir, *resolved_run_dir.parents):
        raise ValueError(f"refusing to reset unexpected candidate run dir: {run_dir}")
    shutil.rmtree(run_dir)


def progressive_candidate_run_dir(args: argparse.Namespace, candidate_version: str) -> Path:
    model_cache_root = Path(getattr(args, "model_cache_root", DEFAULT_MODEL_CACHE_ROOT))
    lifecycle_run_id = str(getattr(args, "lifecycle_run_id", "") or "manual")
    scenario_id = str(getattr(args, "scenario_id", "") or "unknown_scenario")
    return model_cache_root / scenario_id / lifecycle_run_id / candidate_version


def train_candidate_on_trigger(args: argparse.Namespace, decision: LifecycleDecision) -> dict[str, Any]:
    candidate_version = decision.candidate_dataset_version
    if not candidate_version:
        raise ValueError("lifecycle decision did not provide a candidate_dataset_version")
    manifest_path = Path("data/model_datasets") / f"{candidate_version}.csv"
    run_dir = progressive_candidate_run_dir(args, candidate_version)
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
        metric_eval_manifest_path=args.reference_eval_manifest,
        gt_masks_manifest=args.reference_gt_masks_manifest,
        metric_eval_device=args.device,
        metric_eval_every_epochs=1,
        metric_eval_start_epoch=1,
        metric_eval_calibrate_normal=False,
        metric_eval_layer_weights={"layer2": 0.65, "layer3": 0.35},
        metric_eval_apply_score_region_to_map=True,
        metric_eval_profile=args.gate_eval_profile,
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
