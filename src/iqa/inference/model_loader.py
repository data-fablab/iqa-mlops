"""Load production models from the MLflow Registry."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

from iqa.models.artifacts import (
    DEFAULT_ROI_MODEL_VERSION,
    feature_ae_reference_contract_from_manifest,
    validate_feature_ae_reference_manifest,
)
from iqa.models.feature_ae import load_rd_feature_ae_gated
from iqa.promotion import resolve_model_artifacts
from iqa.registry.mlflow_registry import registered_model_name
from iqa.storage.artifacts import (
    resolve_model_artifact_uri,
    sha256_file,
)


@dataclass(frozen=True)
class LoadedModel:
    """Validated immutable production bundle."""

    scenario_id: str
    registered_model_name: str
    version: str
    artifact_uri: str
    model: Any
    feature_ae_version: str = ""
    checkpoint_path: Path | None = None
    decision_thresholds: dict[str, Any] = field(
        default_factory=dict
    )
    reference_contract: Any | None = None
    roi_model_version: str = DEFAULT_ROI_MODEL_VERSION
    model_id: str = ""


class ProdModelLoader:
    """Resolve, validate and atomically expose the prod bundle."""

    def __init__(
        self,
        scenario_id: str,
        *,
        tracking_uri: str | None = None,
    ):
        self.scenario_id = scenario_id
        self.tracking_uri = tracking_uri
        self._loaded_model: LoadedModel | None = None
        self._lock = RLock()

    def load(self) -> LoadedModel:
        model_name = registered_model_name(
            self.scenario_id
        )
        artifact_info = resolve_model_artifacts(
            model_name,
            stage="prod",
            tracking_uri=self.tracking_uri,
        )

        artifact_uri = artifact_info["artifact_uri"]
        registry_version = (
            artifact_info.get("version")
            or self._extract_version_from_uri(
                artifact_uri
            )
        )

        if artifact_uri.startswith("models:/"):
            loaded = self._load_logged_model(
                model_name=model_name,
                model_uri=artifact_uri,
                registry_version=str(registry_version),
            )
        else:
            checkpoint_path = (
                self._resolve_checkpoint_path(
                    artifact_uri
                )
            )
            model = load_rd_feature_ae_gated(
                checkpoint_path
            )
            loaded = LoadedModel(
                scenario_id=self.scenario_id,
                registered_model_name=model_name,
                version=str(registry_version),
                artifact_uri=artifact_uri,
                model=model,
                feature_ae_version=str(
                    registry_version
                ),
                checkpoint_path=checkpoint_path,
            )

        with self._lock:
            self._loaded_model = loaded

        return loaded

    def reload(self) -> LoadedModel:
        return self.load()

    def current(self) -> LoadedModel | None:
        with self._lock:
            return self._loaded_model

    def _load_logged_model(
        self,
        *,
        model_name: str,
        model_uri: str,
        registry_version: str,
    ) -> LoadedModel:
        bundle_root = (
            self._download_logged_model_bundle(
                model_uri
            )
        )
        artifacts_dir = bundle_root / "artifacts"

        checkpoint_path = (
            artifacts_dir / "checkpoint.pt"
        )
        manifest_path = (
            artifacts_dir / "model_manifest.json"
        )
        contract_path = (
            artifacts_dir / "score_contract.json"
        )

        required = (
            bundle_root / "MLmodel",
            checkpoint_path,
            manifest_path,
            contract_path,
        )
        missing = [
            str(path)
            for path in required
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "incomplete_mlflow_feature_ae_bundle: "
                + ", ".join(missing)
            )

        manifest = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )
        feature_ae_version = str(
            manifest.get("model_version") or ""
        )
        if not feature_ae_version:
            raise ValueError(
                "feature_ae_manifest_missing_model_version"
            )

        validate_feature_ae_reference_manifest(
            manifest,
            model_version=feature_ae_version,
        )

        expected_sha256 = str(
            manifest.get("sha256") or ""
        )
        actual_sha256 = sha256_file(
            checkpoint_path
        )
        if not expected_sha256:
            raise ValueError(
                "feature_ae_manifest_missing_sha256"
            )
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "feature_ae_checkpoint_checksum_mismatch: "
                f"expected {expected_sha256}, "
                f"got {actual_sha256}"
            )

        score_contract = json.loads(
            contract_path.read_text(
                encoding="utf-8"
            )
        )
        expected_contract_version = str(
            manifest.get(
                "preprocessing_contract_version"
            )
            or ""
        )
        actual_contract_version = str(
            score_contract.get("version") or ""
        )
        if (
            expected_contract_version
            and actual_contract_version
            != expected_contract_version
        ):
            raise ValueError(
                "feature_ae_score_contract_mismatch: "
                f"expected "
                f"{expected_contract_version!r}, "
                f"got {actual_contract_version!r}"
            )

        reference_contract = (
            feature_ae_reference_contract_from_manifest(
                manifest,
                model_version=feature_ae_version,
            )
        )
        decision_thresholds = dict(
            manifest["decision_thresholds"]
        )
        roi_model_version = str(
            manifest.get("roi_model_version")
            or DEFAULT_ROI_MODEL_VERSION
        )

        model = load_rd_feature_ae_gated(
            checkpoint_path
        )
        model_id = (
            model_uri.removeprefix("models:/")
            .strip("/")
            .split("/", 1)[0]
        )

        return LoadedModel(
            scenario_id=self.scenario_id,
            registered_model_name=model_name,
            version=registry_version,
            artifact_uri=model_uri,
            model=model,
            feature_ae_version=feature_ae_version,
            checkpoint_path=checkpoint_path,
            decision_thresholds=decision_thresholds,
            reference_contract=reference_contract,
            roi_model_version=roi_model_version,
            model_id=model_id,
        )

    def _download_logged_model_bundle(
        self,
        model_uri: str,
    ) -> Path:
        try:
            import mlflow
        except ImportError as error:
            raise ImportError(
                "MLflow is required to download "
                "the production bundle"
            ) from error

        if self.tracking_uri:
            mlflow.set_tracking_uri(
                self.tracking_uri
            )

        cache_root = Path(
            os.environ.get(
                "IQA_MODEL_CACHE_DIR",
                ".cache/iqa/inference_models",
            )
        ).resolve()
        cache_root.mkdir(
            parents=True,
            exist_ok=True,
        )
        destination = Path(
            tempfile.mkdtemp(
                prefix="bundle_",
                dir=cache_root,
            )
        )

        downloaded = (
            mlflow.artifacts.download_artifacts(
                artifact_uri=model_uri,
                dst_path=str(destination),
            )
        )
        return Path(downloaded)

    @staticmethod
    def _extract_version_from_uri(
        artifact_uri: str,
    ) -> str:
        normalized_uri = artifact_uri.replace(
            "\\",
            "/",
        )
        if "mlruns" in normalized_uri:
            parts = normalized_uri.split(
                "mlruns/",
                1,
            )
            if len(parts) > 1:
                remainder = parts[1].split("/")
                if len(remainder) >= 2:
                    return remainder[1]

        if "runs/" in normalized_uri:
            parts = normalized_uri.split(
                "runs/",
                1,
            )
            if len(parts) > 1:
                return parts[1].split("/")[0]

        return "unknown"

    @staticmethod
    def _resolve_checkpoint_path(
        artifact_uri: str,
    ) -> Path:
        return resolve_model_artifact_uri(
            artifact_uri,
            model_version="mlflow_feature_ae",
            filename="checkpoint.pt",
        )


__all__ = [
    "LoadedModel",
    "ProdModelLoader",
]
