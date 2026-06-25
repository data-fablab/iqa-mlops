"""Calibrate the Feature-AE reconstruction decision thresholds (Issue 6).

Twin of ``scripts/calibrate_drift_proxy.py``, but for the image stage of ADR 0010
§3: it runs the **real** ``RealFeatureAEScorer`` over the drift replay plan, where
the baseline checkpoint is trained on ``Casting_class1`` only. The plan replays the
``baseline_domain_class1`` phase (in-distribution) then ``domain_extension_class2/3``
(OOD). The script measures the class1 score distribution, derives ``Orange``/``Red``
(high class1 percentile + margin), verifies the separation (class1 -> anomaly ratio
~0, class2/class3 -> ~1) and emits ``configs/feature_ae_reconstruction_calibration.yaml``
with an HITL block for human validation.

The GPU scoring is isolated behind ``score_plan_phases`` so the derivation /
separation logic is unit-testable without a GPU. Reproducible:
``uv run --extra serving python scripts/calibrate_feature_ae_reconstruction.py \
    --image-root data/raw/hss-iad --write-config``
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import yaml

from iqa.inference.reconstruction_calibration import (
    BASELINE_PHASE,
    DEFAULT_CALIBRATION_PATH,
    DEFAULT_MARGIN,
    DEFAULT_ORANGE_PERCENTILE,
    DEFAULT_RED_PERCENTILE,
    derive_reconstruction_thresholds,
)

DEFAULT_PLAN = Path("data/metadata/casting_flux_replay_plan_drift.csv")
EXTENSION_PHASE_PREFIX = "domain_extension"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--checkpoint", default=None, help="Override the baseline checkpoint path.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--orange-percentile", type=float, default=DEFAULT_ORANGE_PERCENTILE)
    parser.add_argument("--red-percentile", type=float, default=DEFAULT_RED_PERCENTILE)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--max-per-phase", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--write-config", action="store_true", help="Persist the YAML to --output.")
    return parser.parse_args()


def _split_paths(value: str) -> list[str]:
    return [part.strip() for part in value.replace(";", "|").replace(",", "|").split("|") if part.strip()]


def score_plan_phases(
    plan_path: Path,
    image_root: Path,
    score_fn: Callable[[str], float],
    *,
    max_per_phase: int | None = None,
) -> dict[str, list[float]]:
    """Score every plan image with ``score_fn``, grouped by ``scenario_phase``.

    ``score_fn`` takes a local image path and returns the reconstruction score; in
    production it is ``RealFeatureAEScorer.score`` (GPU), but tests inject a stub.
    """
    scores_by_phase: dict[str, list[float]] = defaultdict(list)
    with plan_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            phase = row.get("scenario_phase") or "unknown"
            if max_per_phase is not None and len(scores_by_phase[phase]) >= max_per_phase:
                continue
            for relative_path in _split_paths(row.get("relative_paths") or row.get("relative_path") or ""):
                if max_per_phase is not None and len(scores_by_phase[phase]) >= max_per_phase:
                    break
                scores_by_phase[phase].append(float(score_fn(str(image_root / relative_path))))
    return dict(scores_by_phase)


def _anomaly_rate(scores: list[float], orange: float) -> float:
    if not scores:
        return 0.0
    return round(sum(1 for score in scores if score >= orange) / len(scores), 4)


def calibrate_from_phase_scores(
    scores_by_phase: dict[str, list[float]],
    *,
    baseline_phase: str = BASELINE_PHASE,
    orange_percentile: float = DEFAULT_ORANGE_PERCENTILE,
    red_percentile: float = DEFAULT_RED_PERCENTILE,
    margin: float = DEFAULT_MARGIN,
) -> dict[str, Any]:
    """Derive thresholds from the baseline phase and measure the phase separation."""
    baseline_scores = scores_by_phase.get(baseline_phase) or []
    if not baseline_scores:
        raise ValueError(f"No scores for the baseline phase {baseline_phase!r}; cannot calibrate")

    derivation = derive_reconstruction_thresholds(
        baseline_scores,
        orange_percentile=orange_percentile,
        red_percentile=red_percentile,
        margin=margin,
    )
    orange = derivation["threshold_orange"]
    red = derivation["threshold_red"]

    by_phase: dict[str, dict[str, Any]] = {}
    for phase, scores in sorted(scores_by_phase.items()):
        decisions: Counter[str] = Counter()
        for score in scores:
            decisions["Rouge" if score >= red else ("Orange" if score >= orange else "Vert")] += 1
        by_phase[phase] = {
            "total": len(scores),
            "decisions": dict(decisions),
            "anomaly_rate": _anomaly_rate(scores, orange),
        }

    extension_rates = [
        meta["anomaly_rate"] for phase, meta in by_phase.items()
        if phase.startswith(EXTENSION_PHASE_PREFIX)
    ]
    baseline_rate = by_phase.get(baseline_phase, {}).get("anomaly_rate", 0.0)
    worst_extension = min(extension_rates, default=0.0)
    return {
        "method": derivation["method"],
        "orange_percentile": derivation["orange_percentile"],
        "red_percentile": derivation["red_percentile"],
        "margin": derivation["margin"],
        "thresholds": {
            "threshold_orange": orange,
            "threshold_red": red,
        },
        "class1_score_stats": derivation["class1_score_stats"],
        "by_phase": by_phase,
        "separation": {
            "baseline_phase_anomaly_rate": baseline_rate,
            "extension_phase_anomaly_rate": worst_extension,
            "margin": round(worst_extension - baseline_rate, 4),
        },
    }


def build_calibration_document(calibration: dict[str, Any], *, checkpoint: str | None) -> dict[str, Any]:
    """Wrap the calibration result in the on-disk schema with an HITL block."""
    return {
        "reconstruction_calibration": {
            "checkpoint": checkpoint or "",
            "computed_at": datetime.now(UTC).isoformat(),
            **calibration,
            "hitl": {
                "validated": False,
                "validated_by": None,
                "validated_on": None,
                "note": (
                    "Seuils derives de la phase baseline_domain_class1 (percentile + marge). "
                    "A valider par un humain : verifier class1 -> ratio ~0 et class2/class3 -> ~1, "
                    "puis passer validated a true."
                ),
            },
        }
    }


def main() -> None:
    args = parse_args()
    from iqa.inference.real_inference import RealFeatureAEScorer

    scorer = RealFeatureAEScorer(checkpoint_path=args.checkpoint, device=args.device)
    scores_by_phase = score_plan_phases(
        args.plan, args.image_root, scorer.score, max_per_phase=args.max_per_phase
    )
    calibration = calibrate_from_phase_scores(
        scores_by_phase,
        orange_percentile=args.orange_percentile,
        red_percentile=args.red_percentile,
        margin=args.margin,
    )
    document = build_calibration_document(calibration, checkpoint=scorer.checkpoint_path)

    rendered = yaml.safe_dump(document, sort_keys=False, allow_unicode=True)
    if args.write_config:
        args.output.write_text(rendered, encoding="utf-8")
        print(f"wrote {args.output}")
    print(rendered)


if __name__ == "__main__":
    main()
