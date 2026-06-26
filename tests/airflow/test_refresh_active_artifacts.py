"""Tests for refresh_active_artifacts (Issue 23)."""

from __future__ import annotations

import json

import pytest

from iqa.dags.refresh_active_artifacts import (
    RefreshResult,
    refresh_active_artifacts,
    resolve_new_covered_classes,
)

pytestmark = pytest.mark.unit


def _promoted_summary(checkpoint: str, models: list[str] | None = None) -> dict:
    return {
        "promotion_status": "promoted",
        "candidate_checkpoint": checkpoint,
        "models_promoted": models or [],
        "comparison_history": [],
        "trigger_reason": "drift_confirmed",
        "triggering_class": "Casting_class2",
    }


def _non_promoted_summary() -> dict:
    return {
        "promotion_status": "rejected",
        "candidate_checkpoint": "",
        "models_promoted": [],
        "comparison_history": [],
    }


def test_refresh_copies_checkpoint_on_promotion(tmp_path):
    ckpt = tmp_path / "champion.pt"
    ckpt.write_bytes(b"model-data")
    models_root = tmp_path / "models"

    build_calls = []

    def mock_build(output_dir, covered_classes):
        build_calls.append((output_dir, covered_classes))

    result = refresh_active_artifacts(
        _promoted_summary(str(ckpt)),
        models_root,
        build_bank_fn=mock_build,
    )

    assert result.status == "refreshed"
    assert (models_root / "rd_feature_ae_active" / "checkpoint.pt").read_bytes() == b"model-data"
    assert len(build_calls) == 1


def test_refresh_skips_when_not_promoted(tmp_path):
    result = refresh_active_artifacts(
        _non_promoted_summary(),
        tmp_path / "models",
        build_bank_fn=lambda **kw: None,
    )
    assert result.status == "skipped"
    assert result.reason == "no_promotion"


def test_refresh_derives_covered_classes_from_manifest(tmp_path):
    det_dir = tmp_path / "models" / "patchcore_domain_drift_active"
    det_dir.mkdir(parents=True)
    manifest = {"covered_classes": ["Casting_class1"]}
    (det_dir / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = resolve_new_covered_classes(det_dir, "Casting_class2")
    assert result == ["Casting_class1", "Casting_class2"]


def test_refresh_defaults_class1_when_no_manifest(tmp_path):
    result = resolve_new_covered_classes(tmp_path / "nonexistent", None)
    assert result == ["Casting_class1"]


def test_refresh_builds_bank_with_new_coverage(tmp_path):
    ckpt = tmp_path / "champion.pt"
    ckpt.write_bytes(b"data")
    models_root = tmp_path / "models"
    det_dir = models_root / "patchcore_domain_drift_active"
    det_dir.mkdir(parents=True)
    manifest = {"covered_classes": ["Casting_class1"]}
    (det_dir / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    build_calls = []

    def mock_build(output_dir, covered_classes):
        build_calls.append(covered_classes)

    refresh_active_artifacts(
        _promoted_summary(str(ckpt)),
        models_root,
        build_bank_fn=mock_build,
    )

    assert build_calls[0] == ["Casting_class1", "Casting_class2"]


def test_refresh_result_round_trips():
    result = RefreshResult(
        status="refreshed",
        feature_ae_checkpoint="/a/b.pt",
        detector_dir="/c/d",
        covered_classes=["Casting_class1"],
    )
    d = result.to_dict()
    assert d["status"] == "refreshed"
    assert d["covered_classes"] == ["Casting_class1"]


def test_refresh_idempotent_on_same_summary(tmp_path):
    ckpt = tmp_path / "champion.pt"
    ckpt.write_bytes(b"data")
    models_root = tmp_path / "models"
    summary = _promoted_summary(str(ckpt))

    mock_build = lambda **kw: None

    r1 = refresh_active_artifacts(summary, models_root, build_bank_fn=mock_build)
    r2 = refresh_active_artifacts(summary, models_root, build_bank_fn=mock_build)
    assert r1.status == r2.status == "refreshed"
