"""Reproducibility test for the drift-proxy calibration run (issue 03, critère 5).

Asserts the calibration cleanly separates the in-distribution baseline phase from
the out-of-distribution domain-extension phases, and that the recorded artifact
threshold sits inside that margin.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from scripts.calibrate_drift_proxy import calibrate_drift_proxy

ARTIFACT = Path("configs/drift_proxy_calibration.yaml")


def _calibration() -> dict:
    return calibrate_drift_proxy(argparse.Namespace(
        natural_scenario="production_replay_natural",
        drift_scenario="drift_domain_extension",
        proposed_threshold=0.5,
    ))


def test_natural_regime_has_no_anomalies() -> None:
    assert _calibration()["regimes"]["natural"]["anomaly_rate"] == 0.0


def test_baseline_phase_separates_from_extension_phase() -> None:
    sep = _calibration()["separation"]
    assert sep["drift_baseline_phase_anomaly_rate"] == 0.0
    assert sep["drift_extension_phase_anomaly_rate"] == 1.0
    assert sep["margin"] == 1.0


def test_proposed_threshold_fires_only_on_extension() -> None:
    result = _calibration()
    assert result["fires_on_extension_phase"] is True
    assert result["fires_on_baseline_phase"] is False


def test_recorded_threshold_inside_separation_margin() -> None:
    cfg = yaml.safe_load(ARTIFACT.read_text(encoding="utf-8"))["drift_proxy"]
    sep = _calibration()["separation"]
    threshold = cfg["threshold"]
    assert sep["drift_baseline_phase_anomaly_rate"] < threshold < sep["drift_extension_phase_anomaly_rate"]
    assert cfg["hitl"]["validated"] is True
