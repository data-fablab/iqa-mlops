"""Tests for RealFeatureAEScorer threshold resolution (Issue 6).

The decision thresholds come from the calibration file (no hardcoded constants in
the scorer); resolution order is explicit arg -> calibration file -> env -> fallback.
Constructing the scorer is cheap: the checkpoint loads lazily on score().
"""

from __future__ import annotations

import pytest

from iqa.inference.real_inference import (
    _FALLBACK_THRESHOLD_ORANGE,
    _FALLBACK_THRESHOLD_RED,
    RealFeatureAEScorer,
)

pytestmark = pytest.mark.unit


def _write_calib(path, orange, red):
    path.write_text(
        "reconstruction_calibration:\n"
        f"  thresholds:\n    threshold_orange: {orange}\n    threshold_red: {red}\n",
        encoding="utf-8",
    )
    return str(path)


def test_calibration_file_drives_thresholds(tmp_path, monkeypatch):
    monkeypatch.delenv("IQA_FEATURE_AE_THRESHOLD_ORANGE", raising=False)
    monkeypatch.delenv("IQA_FEATURE_AE_THRESHOLD_RED", raising=False)
    calib = _write_calib(tmp_path / "c.yaml", 0.33, 0.50)
    scorer = RealFeatureAEScorer(calibration_path=calib, device="cpu")
    assert scorer.threshold_orange == 0.33
    assert scorer.threshold_red == 0.50
    assert scorer.threshold_source == "calibration_file"


def test_explicit_args_override_calibration_file(tmp_path):
    calib = _write_calib(tmp_path / "c.yaml", 0.33, 0.50)
    scorer = RealFeatureAEScorer(
        calibration_path=calib, threshold_orange=0.1, threshold_red=0.2, device="cpu"
    )
    assert scorer.threshold_orange == 0.1
    assert scorer.threshold_red == 0.2
    assert scorer.threshold_source == "explicit"


def test_env_used_when_calibration_file_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("IQA_FEATURE_AE_THRESHOLD_ORANGE", "0.07")
    monkeypatch.setenv("IQA_FEATURE_AE_THRESHOLD_RED", "0.11")
    scorer = RealFeatureAEScorer(calibration_path=str(tmp_path / "absent.yaml"), device="cpu")
    assert scorer.threshold_orange == 0.07
    assert scorer.threshold_red == 0.11
    assert scorer.threshold_source == "env_or_fallback"


def test_fallback_constants_when_no_file_and_no_env(tmp_path, monkeypatch):
    monkeypatch.delenv("IQA_FEATURE_AE_THRESHOLD_ORANGE", raising=False)
    monkeypatch.delenv("IQA_FEATURE_AE_THRESHOLD_RED", raising=False)
    scorer = RealFeatureAEScorer(calibration_path=str(tmp_path / "absent.yaml"), device="cpu")
    assert scorer.threshold_orange == _FALLBACK_THRESHOLD_ORANGE
    assert scorer.threshold_red == _FALLBACK_THRESHOLD_RED


def test_decision_uses_resolved_thresholds(tmp_path):
    calib = _write_calib(tmp_path / "c.yaml", 0.30, 0.60)
    scorer = RealFeatureAEScorer(calibration_path=calib, device="cpu")
    assert scorer._decision(0.10) == "Vert"
    assert scorer._decision(0.45) == "Orange"
    assert scorer._decision(0.75) == "Rouge"

class TestPredictHeatmapGating:
    """Predict-time heatmaps are gated (non-Vert), toggleable, and throttled."""

    def test_emit_enabled_default_and_toggle(self, monkeypatch) -> None:
        from iqa.inference.real_inference import emit_predict_heatmaps_enabled

        monkeypatch.delenv("IQA_EMIT_PREDICT_HEATMAPS", raising=False)
        assert emit_predict_heatmaps_enabled() is True
        monkeypatch.setenv("IQA_EMIT_PREDICT_HEATMAPS", "0")
        assert emit_predict_heatmaps_enabled() is False

    def test_throttle_blocks_within_interval(self, monkeypatch) -> None:
        import iqa.inference.real_inference as ri

        monkeypatch.setenv("IQA_PREDICT_HEATMAP_MIN_INTERVAL_S", "60")
        ri._LAST_HEATMAP_TS = 0.0
        assert ri._heatmap_throttle_ready() is True   # first passes
        assert ri._heatmap_throttle_ready() is False  # second within interval blocked

    def test_throttle_disabled_when_interval_zero(self, monkeypatch) -> None:
        import iqa.inference.real_inference as ri

        monkeypatch.setenv("IQA_PREDICT_HEATMAP_MIN_INTERVAL_S", "0")
        assert ri._heatmap_throttle_ready() is True
        assert ri._heatmap_throttle_ready() is True

    def test_vert_never_emits_heatmap(self, tmp_path) -> None:
        # Vert short-circuits before any rendering/upload (no model, no S3 needed).
        scorer = RealFeatureAEScorer(checkpoint_path=str(tmp_path / "ckpt.pt"))
        request = type("R", (), {"scenario_id": "s", "lot_id": "L", "piece_event_id": "p"})()
        assert scorer._maybe_emit_heatmap("img.png", object(), "Vert", request) is None
