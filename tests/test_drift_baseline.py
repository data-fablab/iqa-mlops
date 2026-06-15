"""Tests for versioned drift baselines distinguishing scenario types."""

from __future__ import annotations

from pathlib import Path


from iqa.monitoring.drift_baseline import (
    DriftBaseline,
    DriftBaselineRegistry,
    DriftBaselineStorage,
    DriftQualifier,
)
from iqa.monitoring.lifecycle import LifecycleSignal, should_trigger_lifecycle


class TestDriftBaselineBasic:
    """Tracer bullet: DriftBaseline qualifies drift as expected/unexpected."""

    def test_create_baseline_for_production(self) -> None:
        """Test creating a baseline for production scenario."""
        baseline = DriftBaseline(
            version="v1",
            scenario_type="production",
            teacher_drift_threshold=0.25,
            reconstruction_threshold=0.90,
        )

        assert baseline.version == "v1"
        assert baseline.scenario_type == "production"
        assert baseline.teacher_drift_threshold == 0.25
        assert baseline.reconstruction_threshold == 0.90

    def test_create_baseline_for_extension(self) -> None:
        """Test creating a baseline for domain extension scenario."""
        baseline = DriftBaseline(
            version="v1",
            scenario_type="extension",
            teacher_drift_threshold=0.50,
            reconstruction_threshold=1.20,
        )

        assert baseline.scenario_type == "extension"

    def test_drift_below_threshold_is_expected(self) -> None:
        """Test that drift below threshold is expected for scenario."""
        baseline = DriftBaseline(
            version="v1",
            scenario_type="production",
            teacher_drift_threshold=0.25,
            reconstruction_threshold=0.90,
        )

        # Drift below threshold = expected
        assert baseline.is_expected_drift(teacher_drift=0.15, anomaly_metric=0.80)

    def test_drift_above_threshold_is_unexpected(self) -> None:
        """Test that drift above threshold is unexpected."""
        baseline = DriftBaseline(
            version="v1",
            scenario_type="production",
            teacher_drift_threshold=0.25,
            reconstruction_threshold=0.90,
        )

        # Drift above threshold = unexpected
        assert not baseline.is_expected_drift(teacher_drift=0.30, anomaly_metric=0.80)

    def test_both_metrics_must_be_below_threshold(self) -> None:
        """Test that both metrics must be below threshold."""
        baseline = DriftBaseline(
            version="v1",
            scenario_type="production",
            teacher_drift_threshold=0.25,
            reconstruction_threshold=0.90,
        )

        # One above, one below = unexpected
        assert not baseline.is_expected_drift(teacher_drift=0.20, anomaly_metric=0.95)
        assert not baseline.is_expected_drift(teacher_drift=0.30, anomaly_metric=0.80)

    def test_baseline_to_dict(self) -> None:
        """Test baseline serialization."""
        baseline = DriftBaseline(
            version="v1",
            scenario_type="production",
            teacher_drift_threshold=0.25,
            reconstruction_threshold=0.90,
        )

        data = baseline.to_dict()
        assert data["version"] == "v1"
        assert data["scenario_type"] == "production"
        assert data["teacher_drift_threshold"] == 0.25


class TestDriftBaselineRegistry:
    """Registry for managing versioned baselines."""

    def test_create_production_baseline(self) -> None:
        """Test creating production baseline."""
        registry = DriftBaselineRegistry()

        baseline = registry.create_production_baseline(
            version="v1",
            teacher_drift_threshold=0.25,
            reconstruction_threshold=0.90,
        )

        assert baseline.scenario_type == "production"
        assert baseline.version == "v1"

    def test_create_extension_baseline(self) -> None:
        """Test creating domain extension baseline."""
        registry = DriftBaselineRegistry()

        baseline = registry.create_extension_baseline(
            version="v1",
            teacher_drift_threshold=0.50,
            reconstruction_threshold=1.20,
        )

        assert baseline.scenario_type == "extension"
        assert baseline.version == "v1"

    def test_get_baseline_by_scenario_type(self) -> None:
        """Test retrieving baseline by scenario type."""
        registry = DriftBaselineRegistry()

        baseline_prod = registry.create_production_baseline(
            version="v1",
            teacher_drift_threshold=0.25,
            reconstruction_threshold=0.90,
        )
        registry.register(baseline_prod)

        baseline_ext = registry.create_extension_baseline(
            version="v1",
            teacher_drift_threshold=0.50,
            reconstruction_threshold=1.20,
        )
        registry.register(baseline_ext)

        # Retrieve by type
        assert registry.get_baseline("production").teacher_drift_threshold == 0.25
        assert registry.get_baseline("extension").teacher_drift_threshold == 0.50


class TestDriftQualification:
    """Integration: qualify actual drift as expected/unexpected."""

    def test_production_drift_qualification(self) -> None:
        """Test qualifying drift in production scenario."""
        baseline = DriftBaseline(
            version="v1",
            scenario_type="production",
            teacher_drift_threshold=0.25,
            reconstruction_threshold=0.90,
        )

        # Normal production: small drift expected
        assert baseline.is_expected_drift(teacher_drift=0.15, anomaly_metric=0.75)

        # Production issue: large drift unexpected
        assert not baseline.is_expected_drift(teacher_drift=0.35, anomaly_metric=0.75)

    def test_extension_drift_qualification(self) -> None:
        """Test qualifying drift in extension scenario."""
        baseline = DriftBaseline(
            version="v1",
            scenario_type="extension",
            teacher_drift_threshold=0.50,
            reconstruction_threshold=1.20,
        )

        # Planned extension: larger drift expected
        assert baseline.is_expected_drift(teacher_drift=0.40, anomaly_metric=1.10)

        # Beyond planned extension: unexpected
        assert not baseline.is_expected_drift(teacher_drift=0.60, anomaly_metric=1.10)


class TestDriftQualificationWithMonitoring:
    """Integration: use baseline to qualify drift in monitoring context."""

    def test_qualify_drift_from_scenario_id(self, tmp_path: Path) -> None:
        """Test qualifying drift using scenario_id mapping."""
        # Create baselines
        storage = DriftBaselineStorage(bucket="iqa-baselines", artifact_root=tmp_path)

        baseline_prod = DriftBaseline(
            version="v1",
            scenario_type="production",
            teacher_drift_threshold=0.25,
            reconstruction_threshold=0.90,
        )
        storage.save_baseline(baseline_prod)

        baseline_ext = DriftBaseline(
            version="v1",
            scenario_type="extension",
            teacher_drift_threshold=0.50,
            reconstruction_threshold=1.20,
        )
        storage.save_baseline(baseline_ext)

        # Use qualifier
        qualifier = DriftQualifier(storage)

        # Scenario: production_replay_natural (maps to "production")
        is_expected = qualifier.is_drift_expected(
            scenario_id="production_replay_natural",
            teacher_drift=0.20,
            anomaly_metric=0.85,
        )
        assert is_expected

        # Scenario: drift_domain_extension (maps to "extension")
        is_expected = qualifier.is_drift_expected(
            scenario_id="drift_domain_extension",
            teacher_drift=0.45,
            anomaly_metric=1.10,
        )
        assert is_expected

    def test_detect_unexpected_drift_in_production(self, tmp_path: Path) -> None:
        """Test detecting unexpected drift as signal for lifecycle."""
        storage = DriftBaselineStorage(bucket="iqa-baselines", artifact_root=tmp_path)

        baseline = DriftBaseline(
            version="v1",
            scenario_type="production",
            teacher_drift_threshold=0.25,
            reconstruction_threshold=0.90,
        )
        storage.save_baseline(baseline)

        qualifier = DriftQualifier(storage)

        # Unexpected high drift
        is_expected = qualifier.is_drift_expected(
            scenario_id="production_replay_natural",
            teacher_drift=0.35,
            anomaly_metric=0.85,
        )
        assert not is_expected


class TestDriftBaselineMinIOStorage:
    """Store and load baselines as versioned artifacts in MinIO."""

    def test_save_baseline_to_minio(self, tmp_path: Path) -> None:
        """Test saving baseline to MinIO artifact."""
        baseline = DriftBaseline(
            version="v1",
            scenario_type="production",
            teacher_drift_threshold=0.25,
            reconstruction_threshold=0.90,
        )

        storage = DriftBaselineStorage(bucket="iqa-baselines", artifact_root=tmp_path)

        # Save baseline
        artifact_path = storage.save_baseline(baseline)

        assert artifact_path.exists()
        assert "v1" in artifact_path.name
        assert "production" in artifact_path.name

    def test_load_baseline_from_minio(self, tmp_path: Path) -> None:
        """Test loading baseline from MinIO artifact."""
        baseline = DriftBaseline(
            version="v1",
            scenario_type="production",
            teacher_drift_threshold=0.25,
            reconstruction_threshold=0.90,
        )

        storage = DriftBaselineStorage(bucket="iqa-baselines", artifact_root=tmp_path)

        # Save then load
        storage.save_baseline(baseline)
        loaded = storage.load_baseline("production", version="v1")

        assert loaded.scenario_type == "production"
        assert loaded.version == "v1"
        assert loaded.teacher_drift_threshold == 0.25

    def test_list_baseline_versions(self, tmp_path: Path) -> None:
        """Test listing all versions of a baseline."""
        storage = DriftBaselineStorage(bucket="iqa-baselines", artifact_root=tmp_path)

        # Save multiple versions
        for version in ["v1", "v2", "v3"]:
            baseline = DriftBaseline(
                version=version,
                scenario_type="production",
                teacher_drift_threshold=0.25,
                reconstruction_threshold=0.90,
            )
            storage.save_baseline(baseline)

        versions = storage.list_versions("production")
        assert len(versions) >= 1
        assert "v1" in versions


class TestIntegrationWithLifecycle:
    """Integration with monitoring lifecycle."""

    def test_drift_baseline_informs_lifecycle_decision(self, tmp_path: Path) -> None:
        """Test that drift qualification informs lifecycle trigger.

        Scenario: Production scenario detects unexpected drift.
        Expected: Lifecycle should be triggered for investigation.
        """
        storage = DriftBaselineStorage(bucket="iqa-baselines", artifact_root=tmp_path)

        # Production baseline: tight thresholds (early warning)
        baseline_prod = DriftBaseline(
            version="v1",
            scenario_type="production",
            teacher_drift_threshold=0.25,
            reconstruction_threshold=0.90,
        )
        storage.save_baseline(baseline_prod)

        qualifier = DriftQualifier(storage)

        # Scenario 1: Normal production, drift expected
        is_expected = qualifier.is_drift_expected(
            scenario_id="production_replay_natural",
            teacher_drift=0.15,
            anomaly_metric=0.80,
        )
        assert is_expected

        # Scenario 2: Unexpected drift detected
        is_expected = qualifier.is_drift_expected(
            scenario_id="production_replay_natural",
            teacher_drift=0.30,
            anomaly_metric=0.80,
        )
        assert not is_expected

        # Now lifecycle signal can use this information
        signal = LifecycleSignal(
            scenario_id="production_replay_natural",
            conforming_validated_count=50,
            drift_confirmed=not is_expected,  # Unexpected drift = drift confirmed
        )

        # Trigger lifecycle only if drift is confirmed and other conditions met
        should_trigger = should_trigger_lifecycle(signal)
        assert should_trigger
