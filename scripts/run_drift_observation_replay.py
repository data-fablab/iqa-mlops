"""Replay a natural drift scenario in inference-only mode and emit drift windows."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from scripts import run_replay_lifecycle_cycle as lifecycle
from scripts.airflow_contracts import load_yaml_config, print_json
from scripts.run_monitoring import evaluate_drift_metrics, _push_drift_event


SCENARIO_ID = lifecycle.PIECE_B_TO_PIECE_A_P4_DRIFT_SCENARIO_ID
DEFAULT_OUTPUT_ROOT = Path(".cache/iqa/drift_observation")


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
    parser.add_argument("--thresholds-config", type=Path, default=Path("configs/monitoring_thresholds.yaml"))
    parser.add_argument("--api-url", default=os.getenv("IQA_API_URL", ""))
    parser.add_argument("--service-token", default=os.getenv("IQA_SERVICE_TOKEN", ""))
    parser.add_argument("--require-mlflow-registry", action="store_true")
    parser.add_argument("--initial-classification-registered-model", default=lifecycle.STABLE_PIECE_B_CLASSIFICATION_MODEL_NAME)
    parser.add_argument("--initial-localization-registered-model", default=lifecycle.STABLE_PIECE_B_LOCALIZATION_MODEL_NAME)
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
    summary_path = output_dir / "summary.json"
    thresholds = load_yaml_config(args.thresholds_config)

    classification_runtime, localization_runtime = _resolve_active_runtimes(args)
    roi_checkpoint = lifecycle.resolve_roi_segmenter_checkpoint(lifecycle.DEFAULT_ROI_MODEL_VERSION, strict_checksum=True)
    rows = lifecycle.load_replay_rows(args.scenario_id)
    critical_window_count = 0
    confirmed_once = False
    window = DriftWindow(index=1)
    window_summaries: list[dict[str, object]] = []
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
                    critical_window_count=critical_window_count,
                    confirmed_once=confirmed_once,
                    windows_path=windows_path,
                )
                window_summaries.append(window_summary)
                window = DriftWindow(index=window.index + 1)

    if window.event_count:
        window_summary, critical_window_count, confirmed_once = _finalize_window(
            args,
            window,
            classification_runtime=classification_runtime,
            localization_runtime=localization_runtime,
            thresholds=thresholds,
            critical_window_count=critical_window_count,
            confirmed_once=confirmed_once,
            windows_path=windows_path,
            push_api=False,
        )
        window_summaries.append(window_summary)

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
    summary = {
        "service": "iqa-drift-observation-replay",
        "scenario_id": args.scenario_id,
        "run_id": run_id,
        "status": "validated",
        "events_processed": events_processed,
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
    critical_window_count: int,
    confirmed_once: bool,
    windows_path: Path,
    push_api: bool = True,
) -> tuple[dict[str, object], int, bool]:
    metrics = window.metrics()
    drift_eval = evaluate_drift_metrics(
        scenario_id=args.scenario_id,
        window_events=int(metrics["window_events"]),
        domain_ratio=float(metrics["domain_ratio"]),
        alert_rate=float(metrics["alert_rate"]),
        red_rate=float(metrics["red_rate"]),
        unexpected_red_rate=float(metrics.get("unexpected_red_rate", 0.0)),
        roi_fail_rate=float(metrics["roi_fail_rate"]),
        oracle_fn_rate=float(metrics["oracle_fn_rate"]),
        critical_window_count=critical_window_count,
        drift_confirmed=confirmed_once,
        thresholds=thresholds,
    )
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


def _is_piece_a_p4_event(event: lifecycle.CycleEvent) -> bool:
    return event.scenario_phase in {
        "drift_piece_a_p4_suspected",
        "drift_piece_a_p4_confirmed",
        "correction_replay",
    }


if __name__ == "__main__":
    main()
