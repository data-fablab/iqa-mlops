"""Replay a natural drift scenario in inference-only mode and emit drift windows."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

from scripts import run_replay_lifecycle_cycle as lifecycle
from scripts.airflow_contracts import load_yaml_config, print_json
from scripts.run_monitoring import evaluate_drift_metrics, _push_drift_event


SCENARIO_ID = lifecycle.PIECE_B_TO_PIECE_A_P4_DRIFT_SCENARIO_ID
DEFAULT_OUTPUT_ROOT = Path(".cache/iqa/drift_observation")
ROI_MASK_RESIZE = (64, 64)


@dataclass
class DriftWindow:
    index: int
    events: list[lifecycle.CycleEvent] = field(default_factory=list)

    def append(self, event: lifecycle.CycleEvent) -> None:
        self.events.append(event)

    @property
    def event_count(self) -> int:
        return len(self.events)

    def metrics(self) -> dict[str, float | int]:
        total = len(self.events)
        if total == 0:
            return {
                "window_events": 0,
                "domain_ratio": 0.0,
                "alert_rate": 0.0,
                "red_rate": 0.0,
                "unexpected_red_rate": 0.0,
                "roi_fail_rate": 0.0,
                "oracle_fn_rate": 0.0,
            }
        p4_count = sum(1 for event in self.events if _is_piece_a_p4_event(event))
        alert_count = sum(1 for event in self.events if event.decision.lower() in {"orange", "red"})
        red_count = sum(1 for event in self.events if event.decision.lower() == "red")
        conforming = [event for event in self.events if event.oracle_verdict == "conforme"]
        unexpected_red_count = sum(1 for event in conforming if event.decision.lower() == "red")
        roi_fail_count = sum(1 for event in self.events if event.roi_quality_status.lower() == "fail")
        defective = [event for event in self.events if event.oracle_verdict == "defective"]
        false_negatives = sum(1 for event in defective if event.decision.lower() == "green")
        return {
            "window_events": total,
            "domain_ratio": p4_count / total,
            "alert_rate": alert_count / total,
            "red_rate": red_count / total,
            "unexpected_red_rate": unexpected_red_count / len(conforming) if conforming else 0.0,
            "roi_fail_rate": roi_fail_count / total,
            "oracle_fn_rate": false_negatives / len(defective) if defective else 0.0,
        }

    def phases(self) -> list[str]:
        return sorted({event.scenario_phase for event in self.events if event.scenario_phase})


@dataclass
class RoiMaskReference:
    by_view: dict[str, list[np.ndarray]] = field(default_factory=dict)

    def add_events(self, events: list[lifecycle.CycleEvent]) -> None:
        for event in events:
            vector = _load_roi_mask_vector(event.roi_mask_path)
            if vector is None:
                continue
            self.by_view.setdefault(_roi_view_key(event), []).append(vector)

    def metrics(self, events: list[lifecycle.CycleEvent], *, novelty_distance: float) -> dict[str, float | int]:
        distances: list[float] = []
        area_ratios: list[float] = []
        for event in events:
            area_ratios.append(float(event.roi_ratio or 0.0))
            vector = _load_roi_mask_vector(event.roi_mask_path)
            references = self._references_for(event)
            if vector is None or not references:
                continue
            distances.append(_nearest_jaccard_distance(vector, references))
        if not distances:
            return {
                "roi_mask_nn_distance": 0.0,
                "roi_mask_novelty_rate": 0.0,
                "roi_area_ratio": float(np.median(area_ratios)) if area_ratios else 0.0,
                "roi_mask_reference_count": self.reference_count,
            }
        return {
            "roi_mask_nn_distance": float(np.median(distances)),
            "roi_mask_novelty_rate": sum(1 for distance in distances if distance >= novelty_distance) / len(distances),
            "roi_area_ratio": float(np.median(area_ratios)) if area_ratios else 0.0,
            "roi_mask_reference_count": self.reference_count,
        }

    def _references_for(self, event: lifecycle.CycleEvent) -> list[np.ndarray]:
        view_references = self.by_view.get(_roi_view_key(event), [])
        if view_references:
            return view_references
        return [item for references in self.by_view.values() for item in references]

    @property
    def reference_count(self) -> int:
        return sum(len(references) for references in self.by_view.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-id", default=SCENARIO_ID, choices=sorted(lifecycle.REPLAY_PLANS))
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-cache-root", type=Path, default=lifecycle.DEFAULT_MODEL_CACHE_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-stage", default="test")
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument(
        "--stable-reference-events",
        type=int,
        default=0,
        help="When > 0, replay only this many stable Piece B events before the P4 drift window.",
    )
    parser.add_argument(
        "--drift-observation-windows",
        type=int,
        default=0,
        help="When > 0, replay only this many complete P4 drift windows after the stable reference.",
    )
    parser.add_argument("--thresholds-config", type=Path, default=Path("configs/monitoring_thresholds.yaml"))
    parser.add_argument("--api-url", default=os.getenv("IQA_API_URL", ""))
    parser.add_argument("--service-token", default=os.getenv("IQA_SERVICE_TOKEN", ""))
    parser.add_argument("--require-mlflow-registry", action="store_true")
    parser.add_argument("--initial-classification-registered-model", default=lifecycle.STABLE_PIECE_B_CLASSIFICATION_MODEL_NAME)
    parser.add_argument("--initial-localization-registered-model", default=lifecycle.STABLE_PIECE_B_LOCALIZATION_MODEL_NAME)
    parser.add_argument("--initial-classification-registered-version", default="")
    parser.add_argument("--initial-localization-registered-version", default="")
    return parser.parse_args()


def main() -> None:
    print_json(run_observation(parse_args()))


def run_observation(args: argparse.Namespace) -> dict[str, object]:
    if args.window_size <= 0:
        raise ValueError("--window-size must be > 0")
    run_id = f"drift_observation_{uuid4().hex}"
    output_dir = args.output_root / args.scenario_id / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "events.jsonl"
    windows_path = output_dir / "windows.jsonl"
    drift_context_path = output_dir / "drift_context.json"
    correction_manifest_path = output_dir / "drift_correction_manifest.csv"
    correction_anchor_manifest_path = output_dir / "drift_correction_anchor_piece_b.csv"
    balanced_eval_manifest_path = output_dir / "drift_correction_balanced_eval.csv"
    summary_path = output_dir / "summary.json"
    thresholds = load_yaml_config(args.thresholds_config)
    drift_thresholds = thresholds.get("drift", {}) if isinstance(thresholds, dict) else {}
    novelty_distance = float(drift_thresholds.get("roi_mask_nn_distance_critical", 0.50))

    classification_runtime, localization_runtime = _resolve_active_runtimes(args)
    roi_checkpoint = lifecycle.resolve_roi_segmenter_checkpoint(lifecycle.DEFAULT_ROI_MODEL_VERSION, strict_checksum=True)
    all_rows = lifecycle.load_replay_rows(args.scenario_id)
    rows = _select_observation_rows(
        all_rows,
        stable_reference_events=int(getattr(args, "stable_reference_events", 0) or 0),
        drift_observation_windows=int(getattr(args, "drift_observation_windows", 0) or 0),
        window_size=int(args.window_size),
    )
    critical_window_count = 0
    confirmed_once = False
    window = DriftWindow(index=1)
    roi_reference = RoiMaskReference()
    window_summaries: list[dict[str, object]] = []
    processed_rows: list[dict[str, str]] = []
    processed_events: list[dict[str, object]] = []
    events_processed = 0

    with events_path.open("w", encoding="utf-8") as events_file:
        for row in rows:
            if args.max_events is not None and events_processed >= args.max_events:
                break
            event = lifecycle.process_replay_event(
                row,
                image_root=args.image_root,
                roi_checkpoint=roi_checkpoint,
                feature_checkpoint=classification_runtime.checkpoint,
                decision_thresholds=classification_runtime.decision_thresholds,
                feature_reference_contract=classification_runtime.reference_contract,
                output_dir=output_dir,
                device=args.device,
                visual_store=None,
                active_model_version=classification_runtime.version,
            )
            events_processed += 1
            processed_rows.append(dict(row))
            processed_events.append(event.to_dict())
            events_file.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
            events_file.flush()
            window.append(event)
            if window.event_count >= args.window_size:
                window_summary, critical_window_count, confirmed_once = _finalize_window(
                    args,
                    window,
                    classification_runtime=classification_runtime,
                    localization_runtime=localization_runtime,
                    thresholds=thresholds,
                    roi_reference=roi_reference,
                    novelty_distance=novelty_distance,
                    critical_window_count=critical_window_count,
                    confirmed_once=confirmed_once,
                    windows_path=windows_path,
                )
                window_summaries.append(window_summary)
                roi_reference.add_events([event for event in window.events if not _is_piece_a_p4_event(event)])
                window = DriftWindow(index=window.index + 1)

    if window.event_count:
        window_summary, critical_window_count, confirmed_once = _finalize_window(
            args,
            window,
            classification_runtime=classification_runtime,
            localization_runtime=localization_runtime,
            thresholds=thresholds,
            roi_reference=roi_reference,
            novelty_distance=novelty_distance,
            critical_window_count=critical_window_count,
            confirmed_once=confirmed_once,
            windows_path=windows_path,
            push_api=False,
        )
        window_summaries.append(window_summary)
        roi_reference.add_events([event for event in window.events if not _is_piece_a_p4_event(event)])

    complete_windows = [window for window in window_summaries if window.get("window_complete")]
    final_window = window_summaries[-1] if window_summaries else {}
    last_complete_window = complete_windows[-1] if complete_windows else {}
    ever_suspected = any(bool(window.get("drift_suspected")) for window in complete_windows)
    ever_confirmed = any(bool(window.get("drift_confirmed")) for window in complete_windows)
    first_confirmed_window_index = next(
        (
            window.get("window_index")
            for window in complete_windows
            if bool(window.get("drift_confirmed"))
        ),
        None,
    )
    drift_status = "confirmed" if ever_confirmed else str(last_complete_window.get("status", final_window.get("status", "clear")))
    context_events = [event for event in processed_events if _is_piece_a_p4_payload(event)]
    observed_context_rows = [row for row in processed_rows if _is_piece_a_p4_row(row)]
    context_rows = observed_context_rows
    stable_rows = [row for row in all_rows if _is_stable_piece_b_row(row)]
    _write_correction_manifest(correction_manifest_path, context_rows)
    manifest_balance = _write_balanced_context_manifests(
        anchor_path=correction_anchor_manifest_path,
        eval_path=balanced_eval_manifest_path,
        context_rows=context_rows,
        stable_rows=stable_rows,
    )
    drift_context = _build_drift_context(
        args,
        run_id=run_id,
        drift_context_path=drift_context_path,
        correction_manifest_path=correction_manifest_path,
        correction_anchor_manifest_path=correction_anchor_manifest_path,
        balanced_eval_manifest_path=balanced_eval_manifest_path,
        manifest_balance=manifest_balance,
        context_events=context_events,
        window_summaries=window_summaries,
        classification_runtime=classification_runtime,
        localization_runtime=localization_runtime,
        first_confirmed_window_index=first_confirmed_window_index,
    )
    lifecycle.write_json(drift_context_path, drift_context)
    summary = {
        "service": "iqa-drift-observation-replay",
        "scenario_id": args.scenario_id,
        "run_id": run_id,
        "status": "validated",
        "events_processed": events_processed,
        "events_available": len(all_rows),
        "stable_reference_events": int(getattr(args, "stable_reference_events", 0) or 0),
        "drift_observation_windows": int(getattr(args, "drift_observation_windows", 0) or 0),
        "windows_processed": len(window_summaries),
        "trigger_lifecycle": ever_confirmed,
        "trigger_reason": "drift_piece_a_p4_confirmed" if ever_confirmed else "",
        "drift_confirmed": ever_confirmed,
        "drift_status": drift_status,
        "last_complete_window": last_complete_window,
        "last_complete_window_status": last_complete_window.get("status", ""),
        "ever_suspected": ever_suspected,
        "ever_confirmed": ever_confirmed,
        "first_confirmed_window_index": first_confirmed_window_index,
        "drift_context_path": str(drift_context_path),
        "correction_manifest_path": str(correction_manifest_path),
        "correction_anchor_manifest_path": str(correction_anchor_manifest_path),
        "balanced_eval_manifest_path": str(balanced_eval_manifest_path),
        "manifest_balance": manifest_balance,
        "context_events_total": len(context_events),
        "final_window": final_window,
        "active_classification_runtime": classification_runtime.to_dict(),
        "active_localization_runtime": localization_runtime.to_dict(),
        "events_path": str(events_path),
        "windows_path": str(windows_path),
        "summary_path": str(summary_path),
        "created_at": datetime.now(UTC).isoformat(),
    }
    lifecycle.write_json(summary_path, summary)
    return summary


def _resolve_active_runtimes(
    args: argparse.Namespace,
) -> tuple[lifecycle.ActiveRuntimeModel, lifecycle.ActiveRuntimeModel]:
    fallback = lifecycle.ActiveRuntimeModel(
        version=lifecycle.DEFAULT_FEATURE_AE_MODEL_VERSION,
        checkpoint=lifecycle.resolve_feature_ae_checkpoint(lifecycle.DEFAULT_FEATURE_AE_MODEL_VERSION, strict_checksum=True),
        decision_thresholds=lifecycle.resolve_runtime_thresholds(lifecycle.DEFAULT_FEATURE_AE_MODEL_VERSION),
        reference_contract=lifecycle.load_feature_ae_reference_contract(lifecycle.DEFAULT_FEATURE_AE_MODEL_VERSION),
        registry_model_name=lifecycle.registered_model_name(args.scenario_id),
        registry_stage=args.target_stage,
    )
    classification_runtime = lifecycle.resolve_registered_initial_runtime(
        args,
        model_name=args.initial_classification_registered_model,
        role="classification",
        fallback_thresholds=fallback.decision_thresholds,
    ) or fallback
    localization_runtime = lifecycle.resolve_registered_initial_runtime(
        args,
        model_name=args.initial_localization_registered_model,
        role="localization",
        fallback_thresholds=classification_runtime.decision_thresholds,
    ) or classification_runtime
    return classification_runtime, localization_runtime


def _finalize_window(
    args: argparse.Namespace,
    window: DriftWindow,
    *,
    classification_runtime: lifecycle.ActiveRuntimeModel,
    localization_runtime: lifecycle.ActiveRuntimeModel,
    thresholds: dict[str, object],
    roi_reference: RoiMaskReference,
    novelty_distance: float,
    critical_window_count: int,
    confirmed_once: bool,
    windows_path: Path,
    push_api: bool = True,
) -> tuple[dict[str, object], int, bool]:
    metrics = window.metrics()
    metrics.update(roi_reference.metrics(window.events, novelty_distance=novelty_distance))
    metrics["context_events_total"] = sum(1 for event in window.events if _is_piece_a_p4_event(event))
    metrics["roi_context_complete"] = (
        int(metrics["context_events_total"]) >= int(metrics["window_events"])
        if int(metrics["window_events"]) > 0
        else False
    )
    drift_eval = evaluate_drift_metrics(
        scenario_id=args.scenario_id,
        window_events=int(metrics["window_events"]),
        domain_ratio=float(metrics["domain_ratio"]),
        roi_mask_nn_distance=float(metrics["roi_mask_nn_distance"]),
        roi_mask_novelty_rate=float(metrics["roi_mask_novelty_rate"]),
        roi_area_ratio=float(metrics["roi_area_ratio"]),
        context_events_total=float(metrics["context_events_total"]),
        alert_rate=float(metrics["alert_rate"]),
        red_rate=float(metrics["red_rate"]),
        unexpected_red_rate=float(metrics.get("unexpected_red_rate", 0.0)),
        roi_fail_rate=float(metrics["roi_fail_rate"]),
        oracle_fn_rate=float(metrics["oracle_fn_rate"]),
        critical_window_count=critical_window_count,
        drift_confirmed=confirmed_once,
        thresholds=thresholds,
    )
    if float(metrics["domain_ratio"]) < 0.8 and not confirmed_once:
        drift_eval = dict(drift_eval)
        drift_eval["critical_window"] = False
        drift_eval["drift_confirmed"] = False
        drift_eval["critical_window_count"] = 0
        drift_eval["status"] = "suspected" if drift_eval.get("drift_suspected") else "clear"
    confirmed_once = confirmed_once or bool(drift_eval["drift_confirmed"])
    critical_window_count = int(drift_eval["critical_window_count"])
    summary = {
        "window_index": window.index,
        "scenario_id": args.scenario_id,
        "status": drift_eval["status"],
        "drift_suspected": drift_eval["drift_suspected"],
        "drift_confirmed": drift_eval["drift_confirmed"],
        "critical_window": drift_eval["critical_window"],
        "critical_window_count": critical_window_count,
        "window_complete": push_api,
        "phases": window.phases(),
        "metrics": drift_eval["metrics"],
        "signals": drift_eval["signals"],
        "degradation_signals": drift_eval["degradation_signals"],
        "evaluated_at": datetime.now(UTC).isoformat(),
    }
    lifecycle.append_jsonl(windows_path, summary)
    if push_api:
        _push_drift_event(
            argparse.Namespace(
                api_url=args.api_url,
                service_token=args.service_token,
                scenario_id=args.scenario_id,
                source_domain="piece_a_p4",
                window_events=int(metrics["window_events"]),
                window_index=window.index,
                active_models={
                    "classification": _runtime_metric_identity(classification_runtime),
                    "localization": _runtime_metric_identity(localization_runtime),
                },
            ),
            drift_eval,
        )
    return summary, critical_window_count, confirmed_once


def _runtime_metric_identity(runtime: lifecycle.ActiveRuntimeModel) -> dict[str, str]:
    return {
        "version": runtime.version,
        "registry_model_name": runtime.registry_model_name,
        "registered_model_version": runtime.registered_model_version,
        "registry_stage": runtime.registry_stage,
        "runtime_contract_status": "loaded" if runtime.reference_contract is not None else "default",
    }


def _build_drift_context(
    args: argparse.Namespace,
    *,
    run_id: str,
    drift_context_path: Path,
    correction_manifest_path: Path,
    correction_anchor_manifest_path: Path,
    balanced_eval_manifest_path: Path,
    manifest_balance: dict[str, int],
    context_events: list[dict[str, object]],
    window_summaries: list[dict[str, object]],
    classification_runtime: lifecycle.ActiveRuntimeModel,
    localization_runtime: lifecycle.ActiveRuntimeModel,
    first_confirmed_window_index: object,
) -> dict[str, object]:
    confirmed_window = next(
        (window for window in window_summaries if window.get("window_index") == first_confirmed_window_index),
        {},
    )
    return {
        "schema_version": 1,
        "scenario_id": args.scenario_id,
        "observation_run_id": run_id,
        "confirmed_window_index": first_confirmed_window_index,
        "confirmed_at": confirmed_window.get("evaluated_at", ""),
        "drift_context_path": str(drift_context_path),
        "correction_manifest_path": str(correction_manifest_path),
        "correction_anchor_manifest_path": str(correction_anchor_manifest_path),
        "balanced_eval_manifest_path": str(balanced_eval_manifest_path),
        "manifest_balance": manifest_balance,
        "context_events_total": len(context_events),
        "roi_metrics": confirmed_window.get("metrics", {}),
        "active_models": {
            "classification": _runtime_metric_identity(classification_runtime),
            "localization": _runtime_metric_identity(localization_runtime),
        },
        "retained_events": [
            {
                "event_id": event.get("event_id", ""),
                "piece_event_id": event.get("piece_event_id", ""),
                "scenario_phase": event.get("scenario_phase", ""),
                "source_class": event.get("source_class", ""),
                "roi_mask_path": event.get("roi_mask_path", ""),
                "heatmap_path": event.get("heatmap_path", ""),
                "score": event.get("score", 0.0),
                "decision": event.get("decision", ""),
                "roi_ratio": event.get("roi_ratio", 0.0),
            }
            for event in context_events
        ],
    }


def _write_correction_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    if not fieldnames:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_balanced_context_manifests(
    *,
    anchor_path: Path,
    eval_path: Path,
    context_rows: list[dict[str, str]],
    stable_rows: list[dict[str, str]],
) -> dict[str, int]:
    p4_good_rows = [row for row in context_rows if _is_good_row(row)]
    stable_good_by_bucket = _rows_by_stable_bucket([row for row in stable_rows if _is_good_row(row)])
    stable_all_by_bucket = _rows_by_stable_bucket(stable_rows)
    stable_buckets = _ordered_stable_buckets(stable_all_by_bucket)

    train_anchor_count = len(p4_good_rows)
    anchor_rows: list[dict[str, str]] = []
    for bucket in stable_buckets:
        anchor_rows.extend(stable_good_by_bucket.get(bucket, [])[:train_anchor_count])

    eval_stable_count = len(context_rows)
    eval_stable_rows: list[dict[str, str]] = []
    for bucket in stable_buckets:
        used_for_train = len(stable_good_by_bucket.get(bucket, [])[:train_anchor_count])
        candidates = stable_all_by_bucket.get(bucket, [])
        eval_stable_rows.extend(candidates[used_for_train : used_for_train + eval_stable_count])
    eval_rows = [*context_rows, *eval_stable_rows]

    _write_correction_manifest(anchor_path, anchor_rows)
    _write_correction_manifest(eval_path, eval_rows)
    stats = {
        "train_p4_good_count": len(p4_good_rows),
        "train_piece_b_anchor_count": len(anchor_rows),
        "eval_p4_count": len(context_rows),
        "eval_piece_b_count": len(eval_stable_rows),
        "eval_total_count": len(eval_rows),
    }
    for index, bucket in enumerate(stable_buckets, start=1):
        stats[f"train_p{index}_anchor_count"] = len(stable_good_by_bucket.get(bucket, [])[:train_anchor_count])
        used_for_train = len(stable_good_by_bucket.get(bucket, [])[:train_anchor_count])
        stats[f"eval_p{index}_count"] = len(stable_all_by_bucket.get(bucket, [])[used_for_train : used_for_train + eval_stable_count])
    stats["train_p4_good_count"] = len(p4_good_rows)
    stats["eval_p4_count"] = len(context_rows)
    return stats


def _rows_by_stable_bucket(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    buckets: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        buckets.setdefault(_stable_bucket_key(row), []).append(row)
    return buckets


def _ordered_stable_buckets(rows_by_bucket: dict[str, list[dict[str, str]]]) -> list[str]:
    preferred = ["Casting_class1:1_2", "Casting_class1:1_3", "Casting_class1:2_3"]
    ordered = [bucket for bucket in preferred if bucket in rows_by_bucket]
    ordered.extend(bucket for bucket in sorted(rows_by_bucket) if bucket not in ordered)
    return ordered[:3]


def _stable_bucket_key(row: dict[str, str]) -> str:
    return str(row.get("view_pairs") or row.get("source_class") or "piece_b")


def _select_observation_rows(
    rows: list[dict[str, str]],
    *,
    stable_reference_events: int,
    drift_observation_windows: int,
    window_size: int,
) -> list[dict[str, str]]:
    if stable_reference_events <= 0 and drift_observation_windows <= 0:
        return rows
    stable_rows = [row for row in rows if _is_stable_piece_b_row(row)]
    p4_rows = [row for row in rows if _is_piece_a_p4_row(row)]
    selected_stable = stable_rows[: max(0, stable_reference_events)]
    if drift_observation_windows <= 0:
        selected_p4 = p4_rows
        selected = [*selected_stable, *selected_p4]
    elif drift_observation_windows >= 2:
        mixed_p4_count = max(1, int(round(window_size * 0.20)))
        mixed_stable_count = max(0, window_size - mixed_p4_count)
        mixed_stable = stable_rows[len(selected_stable) : len(selected_stable) + mixed_stable_count]
        mixed_p4 = p4_rows[:mixed_p4_count]
        confirmed_p4 = p4_rows[mixed_p4_count : mixed_p4_count + window_size * (drift_observation_windows - 1)]
        selected_p4 = [*mixed_p4, *confirmed_p4]
        selected = [*selected_stable, *mixed_stable, *mixed_p4, *confirmed_p4]
    else:
        selected_p4 = p4_rows[: drift_observation_windows * window_size]
        selected = [*selected_stable, *selected_p4]
    if not selected_p4:
        raise ValueError("short drift observation selected no P4 rows")
    return selected


def _is_piece_a_p4_event(event: lifecycle.CycleEvent) -> bool:
    return event.scenario_phase in {
        "drift_piece_a_p4_suspected",
        "drift_piece_a_p4_confirmed",
        "correction_replay",
    }


def _is_piece_a_p4_row(row: dict[str, str]) -> bool:
    return str(row.get("scenario_phase") or "") in {
        "drift_piece_a_p4_suspected",
        "drift_piece_a_p4_confirmed",
        "correction_replay",
    }


def _is_stable_piece_b_row(row: dict[str, str]) -> bool:
    return str(row.get("scenario_phase") or "") == "stable_baseline_piece_b"


def _is_good_row(row: dict[str, str]) -> bool:
    label = str(row.get("label") or "").lower()
    oracle = str(row.get("oracle_verdict") or "").lower()
    is_defective = str(row.get("is_defective") or "").lower()
    return label == "good" or oracle == "conforme" or is_defective == "false"


def _is_piece_a_p4_payload(event: dict[str, object]) -> bool:
    return str(event.get("scenario_phase") or "") in {
        "drift_piece_a_p4_suspected",
        "drift_piece_a_p4_confirmed",
        "correction_replay",
    }


def _roi_view_key(event: lifecycle.CycleEvent) -> str:
    stem = Path(event.relative_path).stem
    match = re.search(r"(\d+_\d+)$", stem)
    if match:
        return match.group(1)
    return event.source_class or "unknown"


def _load_roi_mask_vector(path: str) -> np.ndarray | None:
    if not path:
        return None
    mask_path = Path(path)
    if not mask_path.is_file():
        return None
    with Image.open(mask_path) as image:
        mask = image.convert("L").resize(ROI_MASK_RESIZE, Image.Resampling.NEAREST)
        return (np.asarray(mask, dtype=np.uint8) > 127).reshape(-1)


def _nearest_jaccard_distance(vector: np.ndarray, references: list[np.ndarray]) -> float:
    distances: list[float] = []
    for reference in references:
        intersection = np.logical_and(vector, reference).sum()
        union = np.logical_or(vector, reference).sum()
        distances.append(0.0 if union == 0 else 1.0 - float(intersection / union))
    return min(distances) if distances else 0.0


if __name__ == "__main__":
    main()
