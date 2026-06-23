"""Helpers for Marc's production lifecycle dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def aggregate_lots(events: list[dict[str, Any]], *, active_model: str = "") -> list[dict[str, Any]]:
    lots: dict[str, dict[str, Any]] = {}
    for event in events:
        lot_id = str(event.get("lot_id") or "lot_inconnu")
        lot = lots.setdefault(
            lot_id,
            {
                "lot_id": lot_id,
                "pieces": 0,
                "conformes_gt": 0,
                "defauts_gt": 0,
                "vert": 0,
                "orange": 0,
                "rouge": 0,
                "roi_fail_count": 0,
                "model_actif": active_model or "-",
            },
        )
        lot["pieces"] += 1
        oracle = str(event.get("oracle_verdict") or "").lower()
        if oracle in {"defective", "defaut", "defectueux", "non_conforme"}:
            lot["defauts_gt"] += 1
        elif oracle == "conforme":
            lot["conformes_gt"] += 1

        decision = _decision_bucket(event.get("decision"))
        lot[decision] += 1
        if str(event.get("roi_quality_status") or "").lower() not in {"", "ok"}:
            lot["roi_fail_count"] += 1
        event_model = str(event.get("active_model_version") or "")
        if event_model:
            lot["model_actif"] = event_model

    rows = []
    for lot in lots.values():
        pieces = max(int(lot["pieces"]), 1)
        lot["taux_conformite"] = round(100 * int(lot["conformes_gt"]) / pieces, 1)
        lot["roi_fail_rate"] = round(100 * int(lot["roi_fail_count"]) / pieces, 2)
        lot["statut_lot"] = _lot_status(lot)
        rows.append(lot)
    return sorted(rows, key=lambda row: row["lot_id"])


def lifecycle_rows(cycles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for cycle in cycles:
        metrics = cycle.get("metrics") or {}
        stability = cycle.get("candidate_aupimo_stability") or cycle.get("aupimo_stability") or {}
        per_class = cycle.get("candidate_per_class_metrics") or cycle.get("per_class_metrics") or {}
        rows.append(
            {
                "cycle_id": cycle.get("cycle_id"),
                "actif_avant": cycle.get("active_model_before"),
                "modele": cycle.get("candidate_version"),
                "vus": cycle.get("evaluation_seen_events") or cycle.get("seen_events"),
                "defauts_vus": cycle.get("seen_defective"),
                "selected_metric": cycle.get("selected_metric"),
                "selected_epoch": cycle.get("selected_epoch"),
                "selected_value": cycle.get("selected_metric_value"),
                "active_metric_value": cycle.get("active_metric_value"),
                "candidate_metric_value": cycle.get("candidate_metric_value"),
                "metric_delta": cycle.get("metric_delta"),
                "reference_metric_delta": cycle.get("reference_metric_delta"),
                "progressive_metric_delta": cycle.get("progressive_metric_delta"),
                "reference_candidate_metric_value": cycle.get("reference_candidate_metric_value"),
                "progressive_candidate_metric_value": cycle.get("progressive_candidate_metric_value"),
                "promotion_panel_decision": cycle.get("promotion_panel_decision"),
                "per_class_regressions": cycle.get("per_class_regressions"),
                "active_false_negatives": cycle.get("active_false_negatives"),
                "candidate_false_negatives": cycle.get("candidate_false_negatives"),
                "fn_delta": cycle.get("fn_delta"),
                "active_good_red_count": cycle.get("active_good_red_count"),
                "candidate_good_red_count": cycle.get("candidate_good_red_count"),
                "good_red_delta": cycle.get("good_red_delta"),
                "activated_for_next_events": cycle.get("activated_for_next_events"),
                "activation_scope": cycle.get("activation_scope"),
                "cache_status": cycle.get("cache_status"),
                "cache_hit": cycle.get("cache_hit"),
                "pixel_aupimo_1e-5_1e-3": metrics.get("pixel_aupimo_1e-5_1e-3"),
                "pixel_ap": metrics.get("pixel_ap"),
                "image_ap": metrics.get("image_ap"),
                "image_auroc": metrics.get("image_auroc"),
                "image_recall": metrics.get("image_recall"),
                "orange_rate": metrics.get("orange_rate"),
                "alert_rate": metrics.get("alert_rate"),
                "red_rate": metrics.get("red_rate"),
                "good_alert_rate": metrics.get("good_alert_rate"),
                "good_red_rate": metrics.get("good_red_rate"),
                "false_positive_count": metrics.get("false_positive_count"),
                "alert_count": metrics.get("alert_count"),
                "red_count": metrics.get("red_count"),
                "active_good_alert_rate": cycle.get("active_good_alert_rate"),
                "candidate_good_alert_rate": cycle.get("candidate_good_alert_rate"),
                "active_good_red_rate": cycle.get("active_good_red_rate"),
                "candidate_good_red_rate": cycle.get("candidate_good_red_rate"),
                "aupimo_unstable": stability.get("aupimo_unstable"),
                "low_fpr_good_outlier_count": stability.get("low_fpr_good_outlier_count"),
                "max_good_score": stability.get("max_good_score"),
                "max_defect_score": stability.get("max_defect_score"),
                "classes": ", ".join(sorted(per_class)) if isinstance(per_class, dict) else "",
                "gate": cycle.get("gate_decision"),
                "promotion": cycle.get("promotion_status"),
                "stage": cycle.get("registry_stage"),
                "registry": cycle.get("registry_status"),
                "mlflow_run_id": cycle.get("mlflow_run_id"),
                "checkpoint": cycle.get("candidate_checkpoint"),
                "dataset_snapshot_id": cycle.get("dataset_snapshot_id"),
                "calibration_set_id": cycle.get("calibration_set_id"),
                "metric_eval_best_path": cycle.get("metric_eval_best_path"),
            }
        )
    return rows


def production_alerts(lots: list[dict[str, Any]], cycles: list[dict[str, Any]]) -> list[str]:
    alerts: list[str] = []
    for lot in lots:
        if int(lot.get("defauts_gt") or 0) > 0:
            alerts.append(f"{lot['lot_id']} contient {lot['defauts_gt']} defaut(s) GT.")
        if int(lot.get("rouge") or 0) > 0 or int(lot.get("orange") or 0) > 0:
            alerts.append(f"{lot['lot_id']} contient des decisions a verifier/non conformes.")
        if float(lot.get("roi_fail_rate") or 0) > 0:
            alerts.append(f"{lot['lot_id']} a un ROI fail rate de {lot['roi_fail_rate']} %.")
    for cycle in cycles:
        promotion_status = str(cycle.get("promotion_status") or "")
        if promotion_status.startswith("rejected") or cycle.get("gate_decision") == "rejected":
            alerts.append(f"{cycle.get('candidate_version')} rejete par le gate modele.")
        stability = cycle.get("candidate_aupimo_stability") or cycle.get("aupimo_stability") or {}
        if stability.get("aupimo_unstable"):
            alerts.append(f"{cycle.get('cycle_id')} AUPIMO instable : {', '.join(stability.get('unstable_reasons') or [])}.")
    return alerts


def _decision_bucket(value: Any) -> str:
    normalized = str(value or "").lower()
    if normalized in {"green", "vert", "conforme"}:
        return "vert"
    if normalized in {"red", "rouge", "defective"}:
        return "rouge"
    return "orange"


def _lot_status(lot: dict[str, Any]) -> str:
    if int(lot.get("rouge") or 0) > 0 or int(lot.get("defauts_gt") or 0) > 0:
        return "Non conforme"
    if int(lot.get("orange") or 0) > 0 or int(lot.get("roi_fail_count") or 0) > 0:
        return "A verifier"
    return "Conforme"
