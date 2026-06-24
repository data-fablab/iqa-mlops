"""Calibrate the drift-proxy threshold for the chemin B alerting rule (IqaDriftProxy).

Scénario de drift contrôlé : le Feature-AE est entraîné sur le baseline
``Casting_class1``. Le replay ``drift_domain_extension`` rejoue d'abord ce baseline
(phase ``baseline_domain_class1``) puis introduit ``Casting_class2`` puis
``Casting_class3`` (phases ``domain_extension_class*``). Le modèle reconstruit mal
ces domaines OOD -> la décision escalade ``Vert -> Orange -> Rouge``
(cf. ``iqa.inference.drift_scoring``).

Le proxy de drift (HITL, borné [0,1]) est la part d'anomalies (Rouge+Orange) DANS le
régime drift :

    sum(rate(iqa_prediction_total{scenario_id=~"drift.*",decision=~"Orange|Rouge"}[W]))
    / clamp_min(sum(rate(iqa_prediction_total{scenario_id=~"drift.*"}[W])), 1e-9)

Ce script rejoue chaque manifest à travers le scorer réel (sans torch), mesure le
taux d'anomalies par régime ET par phase, documente la séparation et propose un
seuil. Le seuil doit être validé par un humain (point HITL) puis consigné dans
``configs/drift_proxy_calibration.yaml`` pour réutilisation par l'issue 04.

Reproductible : ``uv run --extra serving python scripts/calibrate_drift_proxy.py``
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Any

from iqa.inference.contracts import InferenceRequest, placeholder_inference
from iqa.monitoring.lifecycle import (
    DRIFT_REPLAY_SCENARIO_ID,
    NATURAL_REPLAY_SCENARIO_ID,
)
from iqa.replay.runs import FileBackedReplayRepository

ANOMALY_DECISIONS = ("Orange", "Rouge")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--natural-scenario", default=NATURAL_REPLAY_SCENARIO_ID)
    parser.add_argument("--drift-scenario", default=DRIFT_REPLAY_SCENARIO_ID)
    parser.add_argument(
        "--proposed-threshold", type=float, default=0.5,
        help="Seuil proposé pour le ratio borné [0,1] (validation humaine requise).",
    )
    return parser.parse_args()


def _decision_for(scenario_id: str, row: dict[str, Any]) -> str:
    return placeholder_inference(
        InferenceRequest(
            piece_event_id=row.get("piece_event_id") or row.get("simulated_event_id") or "",
            scenario_id=scenario_id,
            image_uri="memory://calibration",
            source_class=row.get("source_class"),
        )
    ).decision


def _anomaly_rate(decisions: Counter[str], total: int) -> float:
    if total == 0:
        return 0.0
    return round(sum(decisions.get(d, 0) for d in ANOMALY_DECISIONS) / total, 4)


def _regime_summary(repository: FileBackedReplayRepository, scenario_id: str) -> dict[str, Any]:
    events = repository.list_events(scenario_id)
    total = len(events)
    if total == 0:
        raise ValueError(f"Aucun événement de replay pour scenario_id={scenario_id!r}")

    decisions: Counter[str] = Counter()
    by_phase: dict[str, Counter[str]] = defaultdict(Counter)
    for row in events:
        decision = _decision_for(scenario_id, row)
        decisions[decision] += 1
        by_phase[row.get("scenario_phase") or "unknown"][decision] += 1

    phases = {
        phase: {
            "total": sum(counts.values()),
            "decisions": dict(counts),
            "anomaly_rate": _anomaly_rate(counts, sum(counts.values())),
        }
        for phase, counts in sorted(by_phase.items())
    }
    return {
        "scenario_id": scenario_id,
        "total_events": total,
        "decisions": dict(decisions),
        "anomaly_rate": _anomaly_rate(decisions, total),
        "by_phase": phases,
    }


def calibrate_drift_proxy(args: argparse.Namespace) -> dict[str, Any]:
    threshold = args.proposed_threshold
    if not 0.0 < threshold < 1.0:
        raise ValueError("--proposed-threshold doit être dans ]0,1[")

    repository = FileBackedReplayRepository()
    natural = _regime_summary(repository, args.natural_scenario)
    drift = _regime_summary(repository, args.drift_scenario)

    # Le proxy est borné par construction : les phases du régime drift vont d'un
    # taux d'anomalies ~0 (baseline class1, modèle en domaine) à ~1 (extension
    # class2/class3, OOD). La règle tire quand la fenêtre est dominée par l'extension.
    extension_rates = [
        meta["anomaly_rate"]
        for phase, meta in drift["by_phase"].items()
        if phase.startswith("domain_extension")
    ]
    baseline_rates = [
        meta["anomaly_rate"]
        for phase, meta in drift["by_phase"].items()
        if not phase.startswith("domain_extension")
    ]
    worst_extension = min(extension_rates, default=0.0)
    worst_baseline = max(baseline_rates, default=0.0)

    return {
        "proxy_definition": (
            'sum(rate(iqa_prediction_total{scenario_id=~"drift.*",decision=~"Orange|Rouge"}[W])) '
            '/ clamp_min(sum(rate(iqa_prediction_total{scenario_id=~"drift.*"}[W])), 1e-9)'
        ),
        "regimes": {"natural": natural, "drift": drift},
        "separation": {
            "natural_anomaly_rate": natural["anomaly_rate"],
            "drift_baseline_phase_anomaly_rate": worst_baseline,
            "drift_extension_phase_anomaly_rate": worst_extension,
            "margin": round(worst_extension - worst_baseline, 4),
        },
        "proposed_threshold": threshold,
        "fires_on_baseline_phase": worst_baseline >= threshold,
        "fires_on_extension_phase": worst_extension >= threshold,
        "hitl_validated": False,
    }


def main() -> None:
    args = parse_args()
    print(json.dumps(calibrate_drift_proxy(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
