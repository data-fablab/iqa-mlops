"""Tests for the PatchCore domain-drift detector seam (Issue 11).

TDD on the pure parts — max-patch aggregation, seeded coreset subsampling, p90
calibration, regime mapping, and save/load round-trip — without a GPU or the
ImageNet backbone. The bank construction / live scoring is verified against the
GPU stack (verify), not here.
"""

from __future__ import annotations

import pytest
import torch

from iqa.inference.domain_drift import (
    IN_DOMAIN,
    OUT_OF_DOMAIN,
    DomainDriftCalibration,
    PatchCoreDomainDriftDetector,
    calibrate_threshold,
    coreset_subsample,
    max_patch_score,
    regime_for_score,
)

pytestmark = pytest.mark.unit


def test_max_patch_score_is_max_over_patches_of_nn_distance():
    bank = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    # patch A sits on a bank point (NN dist 0); patch B is 3-4-5 away from (0,0).
    patches = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    assert max_patch_score(patches, bank) == pytest.approx(5.0)


def test_max_patch_score_rejects_non_2d():
    with pytest.raises(ValueError):
        max_patch_score(torch.zeros(2, 3, 4), torch.zeros(5, 4))


def test_coreset_subsample_is_deterministic_for_a_seed():
    patches = torch.arange(100 * 3, dtype=torch.float32).reshape(100, 3)
    a = coreset_subsample(patches, 10, seed=42)
    b = coreset_subsample(patches, 10, seed=42)
    c = coreset_subsample(patches, 10, seed=7)
    assert a.shape == (10, 3)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_coreset_subsample_returns_all_when_smaller_than_target():
    patches = torch.zeros(5, 3)
    assert coreset_subsample(patches, 25_000).shape == (5, 3)


def test_calibrate_threshold_is_the_requested_percentile():
    scores = list(range(101))  # 0..100
    assert calibrate_threshold(scores, percentile=90.0) == pytest.approx(90.0)


def test_calibrate_threshold_rejects_empty():
    with pytest.raises(ValueError):
        calibrate_threshold([])


def test_regime_mapping_uses_threshold_inclusively():
    assert regime_for_score(2.0, 3.0) == IN_DOMAIN
    assert regime_for_score(3.0, 3.0) == OUT_OF_DOMAIN
    assert regime_for_score(4.2, 3.0) == OUT_OF_DOMAIN


def test_calibration_round_trips_through_dict():
    calib = DomainDriftCalibration(threshold=3.0, percentile=90.0, class1_score_median=2.6, class1_sample_count=40)
    assert DomainDriftCalibration.from_dict(calib.to_dict()) == calib


def test_save_load_round_trip_preserves_bank_and_calibration(tmp_path):
    bank = torch.randn(64, 8)
    calibration = DomainDriftCalibration(
        threshold=3.1, percentile=90.0, class1_score_median=2.67, class1_sample_count=40
    )
    detector = PatchCoreDomainDriftDetector(bank=bank, calibration=calibration, device="cpu")
    directory = detector.save(tmp_path / "patchcore_domain_drift_v001")

    assert (directory / "memory_bank.pt").exists()
    assert (directory / "calibration.yaml").exists()
    assert (directory / "model_manifest.json").exists()

    reloaded = PatchCoreDomainDriftDetector.load(directory, device="cpu")
    assert reloaded.bank is not None
    assert torch.allclose(reloaded.bank, bank)
    assert reloaded.calibration == calibration


def test_manifest_records_per_piece_threshold_and_purpose(tmp_path):
    bank = torch.randn(32, 4)
    calibration = DomainDriftCalibration(threshold=2.95, percentile=90.0, class1_sample_count=40)
    detector = PatchCoreDomainDriftDetector(bank=bank, calibration=calibration, device="cpu")
    manifest = detector.manifest()
    assert manifest["model_version"] == "patchcore_domain_drift_v001"
    assert manifest["regime_threshold"] == pytest.approx(2.95)
    assert manifest["signal"] == "domain_drift"
    assert "domain_drift_only" in manifest["purpose"]


def test_regime_uses_calibrated_threshold():
    detector = PatchCoreDomainDriftDetector(
        bank=torch.zeros(4, 2), calibration=DomainDriftCalibration(threshold=3.0), device="cpu"
    )
    assert detector.regime(2.5) == IN_DOMAIN
    assert detector.regime(4.2) == OUT_OF_DOMAIN


def test_score_without_bank_raises():
    detector = PatchCoreDomainDriftDetector(device="cpu")
    with pytest.raises(RuntimeError):
        detector.score("/nonexistent.jpg")
