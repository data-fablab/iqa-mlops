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
        rows.append(
            {
                "cycle_id": cycle.get("cycle_id"),
                "actif_avant": cycle.get("active_model_before"),
                "modele": cycle.get("candidate_version"),
                "vus": cycle.get("evaluation_seen_events") or cycle.get("seen_events"),
                "defauts_vus": cycle.get("seen_defective"),
                "selected_metric": cycle.get("selected_metric"),
                "selected_value": cycle.get("selected_metric_value"),
                "active_metric_value": cycle.get("active_metric_value"),
                "candidate_metric_value": cycle.get("candidate_metric_value"),
                "metric_delta": cycle.get("metric_delta"),
                "pixel_aupimo_1e-5_1e-3": metrics.get("pixel_aupimo_1e-5_1e-3"),
                "pixel_ap": metrics.get("pixel_ap"),
                "image_ap": metrics.get("image_ap"),
                "image_auroc": metrics.get("image_auroc"),
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
            alerts.append(f"{lot['lot_id']} contient des decisions orange/rouge.")
        if float(lot.get("roi_fail_rate") or 0) > 0:
            alerts.append(f"{lot['lot_id']} a un ROI fail rate de {lot['roi_fail_rate']} %.")
    for cycle in cycles:
        promotion_status = str(cycle.get("promotion_status") or "")
        if promotion_status.startswith("rejected") or cycle.get("gate_decision") == "rejected":
            alerts.append(f"{cycle.get('candidate_version')} rejete par le gate modele.")
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
        return "A revoir"
    if int(lot.get("orange") or 0) > 0 or int(lot.get("roi_fail_count") or 0) > 0:
        return "Sous surveillance"
    return "Conforme"
