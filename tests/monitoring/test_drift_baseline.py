"""Tests for versioned drift baselines distinguishing scenario types."""

from __future__ import annotations

from pathlib import Path

import pytest

from iqa.monitoring.drift_baseline import (
    DriftBaseline,
    DriftBaselineRegistry,
    DriftBaselineStorage,
    DriftQualifier,
)
from iqa.monitoring.lifecycle import LifecycleSignal, should_trigger_lifecycle


@pytest.fixture
def production_baseline() -> DriftBaseline:
    """Production baseline: tight thresholds (early warning)."""
    return DriftBaseline(
        version="v1",
        scenario_type="production",
        teacher_drift_threshold=0.25,
        reconstruction_threshold=0.90,
    )


@pytest.fixture
def extension_baseline() -> DriftBaseline:
    """Domain-extension baseline: looser thresholds (larger drift planned)."""
    return DriftBaseline(
        version="v1",
        scenario_type="extension",
        teacher_drift_threshold=0.50,
        reconstruction_threshold=1.20,
    )


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

    @pytest.mark.parametrize(
        "teacher_drift, anomaly_metric, expected",
        [
            (0.15, 0.80, True),   # both below threshold
            (0.30, 0.80, False),  # teacher drift above
            (0.20, 0.95, False),  # anomaly metric above
            (0.30, 0.95, False),  # both above
        ],
        ids=["both-below", "teacher-above", "anomaly-above", "both-above"],
    )
    def test_drift_expected_only_when_both_metrics_below(
        self,
        production_baseline: DriftBaseline,
        teacher_drift: float,
        anomaly_metric: float,
        expected: bool,
    ) -> None:
        """Drift is expected only when BOTH metrics stay below their thresholds."""
        result = production_baseline.is_expected_drift(
            teacher_drift=teacher_drift, anomaly_metric=anomaly_metric
        )
        assert bool(result) is expected

    def test_baseline_to_dict(self, production_baseline: DriftBaseline) -> None:
        """Test baseline serialization."""
        data = production_baseline.to_dict()
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

    @pytest.mark.parametrize(
        "baseline_fixture, teacher_drift, anomaly_metric, expected",
        [
            ("production_baseline", 0.15, 0.75, True),   # normal production
            ("production_baseline", 0.35, 0.75, False),  # production issue
            ("extension_baseline", 0.40, 1.10, True),    # planned extension
            ("extension_baseline", 0.60, 1.10, False),   # beyond extension
        ],
        ids=[
            "production-normal",
            "production-issue",
            "extension-planned",
            "extension-beyond",
        ],
    )
    def test_drift_qualification_per_scenario(
        self,
        request: pytest.FixtureRequest,
        baseline_fixture: str,
        teacher_drift: float,
        anomaly_metric: float,
        expected: bool,
    ) -> None:
        """Production (tight) and extension (loose) baselines qualify drift differently."""
        baseline: DriftBaseline = request.getfixturevalue(baseline_fixture)
        result = baseline.is_expected_drift(
            teacher_drift=teacher_drift, anomaly_metric=anomaly_metric
        )
        assert bool(result) is expected


class TestDriftQualificationWithMonitoring:
    """Integration: use baseline to qualify drift in monitoring context."""

    def test_qualify_drift_from_scenario_id(
        self,
        tmp_path: Path,
        production_baseline: DriftBaseline,
        extension_baseline: DriftBaseline,
    ) -> None:
        """Test qualifying drift using scenario_id mapping."""
        # Create baselines
        storage = DriftBaselineStorage(bucket="iqa-baselines", artifact_root=tmp_path)
        storage.save_baseline(production_baseline)
        storage.save_baseline(extension_baseline)

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

    def test_detect_unexpected_drift_in_production(
        self, tmp_path: Path, production_baseline: DriftBaseline
    ) -> None:
        """Test detecting unexpected drift as signal for lifecycle."""
        storage = DriftBaselineStorage(bucket="iqa-baselines", artifact_root=tmp_path)
        storage.save_baseline(production_baseline)

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

    def test_save_baseline_to_minio(
        self, tmp_path: Path, production_baseline: DriftBaseline
    ) -> None:
        """Test saving baseline to MinIO artifact."""
        storage = DriftBaselineStorage(bucket="iqa-baselines", artifact_root=tmp_path)

        # Save baseline
        artifact_path = storage.save_baseline(production_baseline)

        assert artifact_path.exists()
        assert "v1" in artifact_path.name
        assert "production" in artifact_path.name

    def test_load_baseline_from_minio(
        self, tmp_path: Path, production_baseline: DriftBaseline
    ) -> None:
        """Test loading baseline from MinIO artifact."""
        storage = DriftBaselineStorage(bucket="iqa-baselines", artifact_root=tmp_path)

        # Save then load
        storage.save_baseline(production_baseline)
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

    def test_drift_baseline_informs_lifecycle_decision(
        self, tmp_path: Path, production_baseline: DriftBaseline
    ) -> None:
        """Test that drift qualification informs lifecycle trigger.

        Scenario: Production scenario detects unexpected drift.
        Expected: Lifecycle should be triggered for investigation.
        """
        storage = DriftBaselineStorage(bucket="iqa-baselines", artifact_root=tmp_path)

        # Production baseline: tight thresholds (early warning)
        storage.save_baseline(production_baseline)

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
