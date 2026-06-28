"""Tests for the idempotent demo reset (handoff limite #5).

The restore is pure filesystem work, so it is unit-testable against a fake
``models_dir``: it must always re-derive the clean class1-only start from the
baselines, no matter how dirty the active artifacts were left.
"""

from __future__ import annotations

import json

import pytest

from scripts.demo_reset import BASELINE_CLASS, restore_class1_baseline

pytestmark = pytest.mark.unit


def _seed_models(models_dir):
    """Build a baseline + a *dirty* (class1+class2) active layout."""
    pc_base = models_dir / "patchcore_domain_drift_v001"
    pc_base.mkdir(parents=True)
    (pc_base / "memory_bank.pt").write_bytes(b"class1-bank")
    (pc_base / "calibration.yaml").write_text("threshold: 3.6\n", encoding="utf-8")
    (pc_base / "class_scores.csv").write_text("score\n2.6\n", encoding="utf-8")
    (pc_base / "model_manifest.json").write_text(
        json.dumps({"model_version": "patchcore_domain_drift_v001", "threshold": 3.6}),
        encoding="utf-8",
    )

    pc_active = models_dir / "patchcore_domain_drift_active"
    pc_active.mkdir(parents=True)
    (pc_active / "memory_bank.pt").write_bytes(b"class1+class2-bank")
    (pc_active / "model_manifest.json").write_text(
        json.dumps({"covered_classes": ["Casting_class1", "Casting_class2"]}),
        encoding="utf-8",
    )

    ae_base = models_dir / "rd_feature_ae_class1_baseline"
    ae_base.mkdir(parents=True)
    (ae_base / "checkpoint.pt").write_bytes(b"ae-class1")

    ae_active = models_dir / "rd_feature_ae_active"
    ae_active.mkdir(parents=True)
    (ae_active / "checkpoint.pt").write_bytes(b"ae-class1+class2")


def test_restore_pins_covered_classes_to_class1(tmp_path):
    _seed_models(tmp_path)
    drift_state = tmp_path / "drift" / "state.json"

    restore_class1_baseline(tmp_path, drift_state)

    manifest = json.loads(
        (tmp_path / "patchcore_domain_drift_active" / "model_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["covered_classes"] == [BASELINE_CLASS]


def test_restore_copies_baseline_bank_and_checkpoint(tmp_path):
    _seed_models(tmp_path)
    drift_state = tmp_path / "drift" / "state.json"

    restore_class1_baseline(tmp_path, drift_state)

    bank = (tmp_path / "patchcore_domain_drift_active" / "memory_bank.pt").read_bytes()
    ckpt = (tmp_path / "rd_feature_ae_active" / "checkpoint.pt").read_bytes()
    assert bank == b"class1-bank"  # baseline overwrote the dirty class1+class2 bank
    assert ckpt == b"ae-class1"


def test_restore_writes_drift_state_class1_only(tmp_path):
    _seed_models(tmp_path)
    drift_state = tmp_path / "drift" / "state.json"

    restore_class1_baseline(tmp_path, drift_state)

    state = json.loads(drift_state.read_text(encoding="utf-8"))
    assert state == {"classes": {BASELINE_CLASS: "covered"}}


def test_restore_is_idempotent(tmp_path):
    _seed_models(tmp_path)
    drift_state = tmp_path / "drift" / "state.json"

    first = restore_class1_baseline(tmp_path, drift_state).to_dict()
    second = restore_class1_baseline(tmp_path, drift_state).to_dict()
    assert first == second
    manifest = json.loads(
        (tmp_path / "patchcore_domain_drift_active" / "model_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["covered_classes"] == [BASELINE_CLASS]


def test_restore_tolerates_bom_in_baseline_manifest(tmp_path):
    _seed_models(tmp_path)
    # A PowerShell-edited manifest may carry a UTF-8 BOM (handoff piège #4).
    pc_base = tmp_path / "patchcore_domain_drift_v001" / "model_manifest.json"
    pc_base.write_bytes(b"\xef\xbb\xbf" + json.dumps({"threshold": 3.6}).encode("utf-8"))
    drift_state = tmp_path / "drift" / "state.json"

    result = restore_class1_baseline(tmp_path, drift_state)
    assert result.covered_classes == [BASELINE_CLASS]


def test_restore_raises_when_baseline_absent(tmp_path):
    # Only the active dirs exist, no baselines -> must fail loudly, not silently.
    (tmp_path / "patchcore_domain_drift_active").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        restore_class1_baseline(tmp_path, tmp_path / "drift" / "state.json")
