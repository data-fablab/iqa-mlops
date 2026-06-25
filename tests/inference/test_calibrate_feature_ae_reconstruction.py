"""Tests for the reconstruction calibration script logic (Issue 6).

Derivation from the baseline phase + per-phase separation are pure; the GPU scoring
(score_plan_phases with the real scorer) is validated by verify. score_plan_phases
is exercised with a stub score_fn over a tiny plan CSV.
"""

from __future__ import annotations

import pytest

from scripts.calibrate_feature_ae_reconstruction import (
    build_calibration_document,
    calibrate_from_phase_scores,
    score_plan_phases,
)

pytestmark = pytest.mark.unit


def test_separation_class1_quiet_extension_anomalous() -> None:
    scores_by_phase = {
        # In-distribution class1: low, tight scores -> all Vert.
        "baseline_domain_class1": [0.01, 0.02, 0.015, 0.018, 0.02],
        # OOD class2/class3: high scores -> all flagged.
        "domain_extension_class2": [0.5, 0.6, 0.55],
        "domain_extension_class3": [0.7, 0.8, 0.75],
    }
    result = calibrate_from_phase_scores(scores_by_phase, margin=0.05)

    assert result["separation"]["baseline_phase_anomaly_rate"] == 0.0
    assert result["separation"]["extension_phase_anomaly_rate"] == 1.0
    assert result["separation"]["margin"] == 1.0
    assert result["thresholds"]["threshold_orange"] <= result["thresholds"]["threshold_red"]
    assert result["by_phase"]["domain_extension_class2"]["decisions"].get("Vert", 0) == 0


def test_missing_baseline_phase_raises() -> None:
    with pytest.raises(ValueError, match="baseline"):
        calibrate_from_phase_scores({"domain_extension_class2": [0.5]})


def test_build_document_has_thresholds_and_hitl_block() -> None:
    calibration = calibrate_from_phase_scores(
        {"baseline_domain_class1": [0.01, 0.02], "domain_extension_class2": [0.9]}
    )
    document = build_calibration_document(calibration, checkpoint="/opt/iqa/models/x/checkpoint.pt")
    block = document["reconstruction_calibration"]
    assert block["checkpoint"] == "/opt/iqa/models/x/checkpoint.pt"
    assert "threshold_orange" in block["thresholds"]
    assert block["hitl"]["validated"] is False


def test_score_plan_phases_groups_by_phase(tmp_path) -> None:
    plan = tmp_path / "plan.csv"
    plan.write_text(
        "scenario_phase,relative_paths\n"
        "baseline_domain_class1,class1/a.jpg\n"
        "baseline_domain_class1,class1/b.jpg\n"
        "domain_extension_class2,class2/c.jpg\n",
        encoding="utf-8",
    )
    # Stub scorer: score = length of the path (deterministic, no GPU).
    scores = score_plan_phases(plan, tmp_path, lambda p: float(len(p)))

    assert len(scores["baseline_domain_class1"]) == 2
    assert len(scores["domain_extension_class2"]) == 1


def test_score_plan_phases_respects_max_per_phase(tmp_path) -> None:
    plan = tmp_path / "plan.csv"
    plan.write_text(
        "scenario_phase,relative_paths\n"
        "baseline_domain_class1,class1/a.jpg\n"
        "baseline_domain_class1,class1/b.jpg\n"
        "baseline_domain_class1,class1/c.jpg\n",
        encoding="utf-8",
    )
    scores = score_plan_phases(plan, tmp_path, lambda p: 1.0, max_per_phase=2)
    assert len(scores["baseline_domain_class1"]) == 2