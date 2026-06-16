"""Versioned drift baselines distinguishing scenario types."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DriftBaseline:
    """Versioned baseline for drift qualification by scenario type."""

    version: str
    scenario_type: str  # "production" or "extension"
    teacher_drift_threshold: float
    reconstruction_threshold: float

    def is_expected_drift(
        self,
        teacher_drift: float,
        anomaly_metric: float,
    ) -> bool:
        """Check if observed drift is expected for this scenario.

        Args:
            teacher_drift: Teacher model drift metric
            anomaly_metric: Reconstruction/anomaly metric

        Returns:
            True if drift is within expected thresholds for scenario_type
        """
        return (
            teacher_drift <= self.teacher_drift_threshold
            and anomaly_metric <= self.reconstruction_threshold
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize baseline to dict."""
        return asdict(self)


class DriftBaselineRegistry:
    """Registry for managing versioned drift baselines."""

    def __init__(self) -> None:
        """Initialize registry."""
        self._baselines: dict[str, DriftBaseline] = {}

    @staticmethod
    def _create_baseline(
        scenario_type: str,
        version: str,
        teacher_drift_threshold: float,
        reconstruction_threshold: float,
    ) -> DriftBaseline:
        return DriftBaseline(
            version=version,
            scenario_type=scenario_type,
            teacher_drift_threshold=teacher_drift_threshold,
            reconstruction_threshold=reconstruction_threshold,
        )

    def create_production_baseline(
        self,
        version: str,
        teacher_drift_threshold: float,
        reconstruction_threshold: float,
    ) -> DriftBaseline:
        """Create baseline for production scenario.

        Args:
            version: Baseline version identifier
            teacher_drift_threshold: Max teacher drift for production
            reconstruction_threshold: Max reconstruction error

        Returns:
            DriftBaseline instance
        """
        return self._create_baseline(
            "production", version, teacher_drift_threshold, reconstruction_threshold
        )

    def create_extension_baseline(
        self,
        version: str,
        teacher_drift_threshold: float,
        reconstruction_threshold: float,
    ) -> DriftBaseline:
        """Create baseline for domain extension scenario.

        Args:
            version: Baseline version identifier
            teacher_drift_threshold: Max teacher drift for extension
            reconstruction_threshold: Max reconstruction error

        Returns:
            DriftBaseline instance
        """
        return self._create_baseline(
            "extension", version, teacher_drift_threshold, reconstruction_threshold
        )

    def register(self, baseline: DriftBaseline) -> None:
        """Register baseline in registry.

        Args:
            baseline: DriftBaseline to register
        """
        self._baselines[baseline.scenario_type] = baseline

    def get_baseline(self, scenario_type: str) -> DriftBaseline:
        """Get registered baseline for scenario type.

        Args:
            scenario_type: "production" or "extension"

        Returns:
            DriftBaseline for scenario

        Raises:
            KeyError: if scenario_type not registered
        """
        if scenario_type not in self._baselines:
            raise KeyError(f"No baseline registered for scenario_type: {scenario_type}")
        return self._baselines[scenario_type]


class DriftBaselineStorage:
    """Store and load baselines as versioned artifacts."""

    def __init__(self, bucket: str = "iqa-baselines", artifact_root: Path | None = None) -> None:
        """Initialize baseline storage.

        Args:
            bucket: MinIO bucket name (default: iqa-baselines)
            artifact_root: Local root for artifact storage (for testing)
        """
        self.bucket = bucket
        self.artifact_root = artifact_root or Path.home() / ".cache" / "iqa-baselines"
        self.artifact_root.mkdir(parents=True, exist_ok=True)

    def _get_baseline_path(self, scenario_type: str, version: str) -> Path:
        """Get artifact path for baseline."""
        return self.artifact_root / f"drift_baseline_{scenario_type}_{version}.json"

    def save_baseline(self, baseline: DriftBaseline) -> Path:
        """Save baseline as artifact.

        Args:
            baseline: DriftBaseline to save

        Returns:
            Path to saved artifact
        """
        artifact_path = self._get_baseline_path(baseline.scenario_type, baseline.version)
        artifact_path.write_text(json.dumps(baseline.to_dict(), indent=2))
        return artifact_path

    def load_baseline(self, scenario_type: str, version: str) -> DriftBaseline:
        """Load baseline from artifact.

        Args:
            scenario_type: "production" or "extension"
            version: Baseline version

        Returns:
            DriftBaseline instance

        Raises:
            FileNotFoundError: if artifact not found
        """
        artifact_path = self._get_baseline_path(scenario_type, version)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Baseline artifact not found: {artifact_path}")

        data = json.loads(artifact_path.read_text())
        return DriftBaseline(**data)

    def list_versions(self, scenario_type: str) -> list[str]:
        """List all versions for a scenario type.

        Args:
            scenario_type: "production" or "extension"

        Returns:
            List of version identifiers
        """
        versions = []
        pattern = f"drift_baseline_{scenario_type}_*.json"
        for path in self.artifact_root.glob(pattern):
            # Extract version from filename
            version = path.stem.replace(f"drift_baseline_{scenario_type}_", "")
            versions.append(version)
        return sorted(versions)


class DriftQualifier:
    """Qualify actual drift as expected/unexpected using baselines."""

    # Map scenario_id to baseline scenario_type
    SCENARIO_TYPE_MAP = {
        "production_replay_natural": "production",
        "drift_domain_extension": "extension",
    }

    def __init__(self, storage: DriftBaselineStorage) -> None:
        """Initialize qualifier with baseline storage.

        Args:
            storage: DriftBaselineStorage for loading baselines
        """
        self.storage = storage

    def is_drift_expected(
        self,
        scenario_id: str,
        teacher_drift: float,
        anomaly_metric: float,
        version: str = "v1",
    ) -> bool:
        """Check if drift is expected for scenario.

        Args:
            scenario_id: Scenario identifier (production_replay_natural or drift_domain_extension)
            teacher_drift: Teacher model drift metric
            anomaly_metric: Reconstruction/anomaly metric
            version: Baseline version to use

        Returns:
            True if drift is within expected thresholds for scenario

        Raises:
            ValueError: if scenario_id not recognized
        """
        if scenario_id not in self.SCENARIO_TYPE_MAP:
            raise ValueError(f"Unknown scenario_id: {scenario_id}")

        scenario_type = self.SCENARIO_TYPE_MAP[scenario_id]

        try:
            baseline = self.storage.load_baseline(scenario_type, version=version)
        except FileNotFoundError:
            # If no baseline exists, assume drift is unexpected (fail-safe)
            return False

        return baseline.is_expected_drift(teacher_drift, anomaly_metric)


__all__ = ["DriftBaseline", "DriftBaselineRegistry", "DriftBaselineStorage", "DriftQualifier"]
