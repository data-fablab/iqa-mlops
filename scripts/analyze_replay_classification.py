"""Analyze replay classification performance against oracle GT."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


ALERT_DECISIONS = {"orange", "red", "rouge", "defective", "non_conforme", "non conforme"}
GREEN_DECISIONS = {"green", "vert", "conforme"}
DEFECT_ORACLES = {"defective", "defaut", "defectueux", "non_conforme", "non conforme"}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_defective(event: dict[str, Any]) -> bool:
    return str(event.get("oracle_verdict") or "").lower() in DEFECT_ORACLES


def is_alert(event: dict[str, Any]) -> bool:
    return str(event.get("decision") or "").lower() in ALERT_DECISIONS


def score(event: dict[str, Any]) -> float | None:
    value = event.get("score")
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def threshold(event: dict[str, Any], name: str) -> float | None:
    value = event.get(name)
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[int(pos)]
    return ordered[lower] * (upper - pos) + ordered[upper] * (pos - lower)


def score_summary(events: Iterable[dict[str, Any]]) -> dict[str, float | int | None]:
    values = [s for event in events if (s := score(event)) is not None]
    if not values:
        return {"n": 0, "min": None, "p25": None, "median": None, "p75": None, "p95": None, "max": None, "mean": None}
    return {
        "n": len(values),
        "min": min(values),
        "p25": quantile(values, 0.25),
        "median": quantile(values, 0.5),
        "p75": quantile(values, 0.75),
        "p95": quantile(values, 0.95),
        "max": max(values),
        "mean": mean(values),
    }


def confusion(events: list[dict[str, Any]]) -> dict[str, float | int]:
    tp = sum(1 for event in events if is_defective(event) and is_alert(event))
    fn = sum(1 for event in events if is_defective(event) and not is_alert(event))
    fp = sum(1 for event in events if not is_defective(event) and is_alert(event))
    tn = sum(1 for event in events if not is_defective(event) and not is_alert(event))
    defects = tp + fn
    goods = tn + fp
    alerts = tp + fp
    return {
        "pieces": len(events),
        "oracle_good": goods,
        "oracle_defect": defects,
        "tp_defect_detected": tp,
        "fn_missed_defect": fn,
        "fp_good_alerted": fp,
        "tn_good_accepted": tn,
        "defect_recall": tp / defects if defects else 0.0,
        "alert_precision": tp / alerts if alerts else 0.0,
        "false_positive_rate": fp / goods if goods else 0.0,
    }


def threshold_at_allowed_good_alerts(events: list[dict[str, Any]], allowed_good_alerts: int) -> dict[str, float | int | None]:
    goods = sorted((score(event) for event in events if not is_defective(event) and score(event) is not None), reverse=True)
    defects = [score(event) for event in events if is_defective(event) and score(event) is not None]
    defects = [value for value in defects if value is not None]
    if not goods and not defects:
        return {"allowed_good_alerts": allowed_good_alerts, "threshold": None, "detected": 0, "fn": 0, "recall": 0.0}
    if allowed_good_alerts <= 0:
        threshold_value = (goods[0] + 1e-9) if goods else min(defects)
    elif allowed_good_alerts >= len(goods):
        threshold_value = min(goods) if goods else min(defects)
    else:
        threshold_value = goods[allowed_good_alerts - 1]
    detected = sum(1 for value in defects if value >= threshold_value)
    total_defects = len(defects)
    return {
        "allowed_good_alerts": allowed_good_alerts,
        "threshold": threshold_value,
        "detected": detected,
        "fn": total_defects - detected,
        "recall": detected / total_defects if total_defects else 0.0,
    }


def group_by(events: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        groups[str(event.get(key) or "-")].append(event)
    return dict(sorted(groups.items()))


def format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def print_table(title: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    print(f"\n== {title} ==")
    if not rows:
        print("(empty)")
        return
    widths = {
        column: max(len(column), *(len(format_value(row.get(column))) for row in rows))
        for column in columns
    }
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(format_value(row.get(column)).ljust(widths[column]) for column in columns))


def analyze_run(run_dir: Path) -> None:
    events = read_jsonl(run_dir / "events.jsonl")
    cycles = read_jsonl(run_dir / "cycles.jsonl") if (run_dir / "cycles.jsonl").exists() else []
    progress = read_json(run_dir / "progress.json")

    print(f"run_dir: {run_dir}")
    print(f"phase: {progress.get('phase')}")
    print(f"events: {len(events)}")
    print(f"active_model_final: {progress.get('active_model_final') or progress.get('active_model_version')}")
    print(f"best_cycle: {progress.get('best_cycle')} best_metric_value: {progress.get('best_metric_value')}")

    print_table(
        "confusion modele vs oracle Sophie",
        [confusion(events)],
        [
            "pieces",
            "oracle_good",
            "oracle_defect",
            "tp_defect_detected",
            "fn_missed_defect",
            "fp_good_alerted",
            "tn_good_accepted",
            "defect_recall",
            "alert_precision",
            "false_positive_rate",
        ],
    )

    score_rows = []
    for label, subset in [
        ("oracle_good", [event for event in events if not is_defective(event)]),
        ("oracle_defect", [event for event in events if is_defective(event)]),
    ]:
        row = {"group": label}
        row.update(score_summary(subset))
        score_rows.append(row)
    print_table("distribution des scores replay", score_rows, ["group", "n", "min", "p25", "median", "p75", "p95", "max", "mean"])

    threshold_rows = []
    for model, subset in group_by(events, "active_model_version").items():
        orange_values = [value for event in subset if (value := threshold(event, "threshold_orange")) is not None]
        red_values = [value for event in subset if (value := threshold(event, "threshold_red")) is not None]
        row = {"active_model_version": model}
        row.update(confusion(subset))
        row["score_good_max"] = score_summary([event for event in subset if not is_defective(event)])["max"]
        row["score_defect_max"] = score_summary([event for event in subset if is_defective(event)])["max"]
        row["threshold_orange_median"] = quantile(orange_values, 0.5)
        row["threshold_red_median"] = quantile(red_values, 0.5)
        threshold_rows.append(row)
    print_table(
        "par modele actif sur le replay",
        threshold_rows,
        [
            "active_model_version",
            "pieces",
            "oracle_defect",
            "tp_defect_detected",
            "fn_missed_defect",
            "fp_good_alerted",
            "defect_recall",
            "alert_precision",
            "score_good_max",
            "score_defect_max",
            "threshold_orange_median",
            "threshold_red_median",
        ],
    )

    print_table(
        "seuils alternatifs sur tout le replay",
        [threshold_at_allowed_good_alerts(events, allowed) for allowed in [0, 1, 2, 5, 10, 20, 50]],
        ["allowed_good_alerts", "threshold", "detected", "fn", "recall"],
    )

    cycle_rows = []
    for cycle in cycles:
        cycle_rows.append(
            {
                "cycle_id": cycle.get("cycle_id"),
                "gate_decision": cycle.get("gate_decision"),
                "promotion_status": cycle.get("promotion_status"),
                "active_metric": cycle.get("active_metric_value"),
                "candidate_metric": cycle.get("candidate_metric_value"),
                "metric_delta": cycle.get("metric_delta"),
                "active_fn": cycle.get("active_false_negatives"),
                "candidate_fn": cycle.get("candidate_false_negatives"),
                "candidate_recall": (cycle.get("classification_gate") or {}).get("candidate_image_recall"),
            }
        )
    print_table(
        "cycles gate reference",
        cycle_rows,
        [
            "cycle_id",
            "gate_decision",
            "promotion_status",
            "active_metric",
            "candidate_metric",
            "metric_delta",
            "active_fn",
            "candidate_fn",
            "candidate_recall",
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, help="Replay lifecycle run directory")
    args = parser.parse_args()
    analyze_run(args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
