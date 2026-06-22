"""Model-version helpers resolving checkpoint manifests to local cache paths."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from iqa.storage.artifacts import resolve_model_artifact_from_manifest

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_MANIFESTS_DIR = DEFAULT_REPO_ROOT / "models" / "manifests"
DEFAULT_ROI_MODEL_VERSION = "roi_segmenter_v001_fixed"
DEFAULT_FEATURE_AE_MODEL_VERSION = "rd_feature_ae_gated_v001_bootstrap"
FEATURE_AE_CHAMPION_CONTRACT_VERSION = "feature_ae_champion_v001"
FEATURE_AE_CHAMPION_REQUIRED_FIELDS = {
    "version",
    "teacher_weights",
    "layers",
    "layer_weights",
    "roi_mode",
    "roi_threshold",
    "score_smoothing",
    "score_image",
    "topk_fraction",
    "tile_size",
    "context_size",
    "tile_stride",
}


def model_manifest_path(model_version: str) -> Path:
    configured_repo_root = os.environ.get("IQA_REPO_ROOT")
    manifests_dir = (
        Path(configured_repo_root) / "models" / "manifests"
        if configured_repo_root
        else MODEL_MANIFESTS_DIR
    )
    return manifests_dir / model_version / "model_manifest.json"


def load_model_manifest(model_version: str) -> dict[str, Any]:
    path = model_manifest_path(model_version)
    return json.loads(path.read_text(encoding="utf-8"))


def load_feature_ae_decision_thresholds(
    model_version: str = DEFAULT_FEATURE_AE_MODEL_VERSION,
) -> dict[str, Any] | None:
    manifest = load_model_manifest(model_version)
    validate_feature_ae_champion_manifest(manifest, model_version=model_version)
    thresholds = manifest.get("decision_thresholds")
    if not isinstance(thresholds, dict):
        return None
    return thresholds


def validate_feature_ae_champion_manifest(
    manifest: dict[str, Any],
    *,
    model_version: str = DEFAULT_FEATURE_AE_MODEL_VERSION,
) -> None:
    contract = manifest.get("feature_ae_champion_contract")
    if not isinstance(contract, dict):
        raise ValueError(f"Feature-AE model {model_version!r} is missing feature_ae_champion_contract")
    missing = sorted(FEATURE_AE_CHAMPION_REQUIRED_FIELDS - set(contract))
    if missing:
        raise ValueError(
            f"Feature-AE model {model_version!r} has incomplete champion contract: missing {', '.join(missing)}"
        )
    if contract.get("version") != FEATURE_AE_CHAMPION_CONTRACT_VERSION:
        raise ValueError(
            f"Feature-AE model {model_version!r} uses unsupported score contract {contract.get('version')!r}"
        )
    thresholds = manifest.get("decision_thresholds")
    if isinstance(thresholds, dict) and thresholds.get("score_contract_version") != FEATURE_AE_CHAMPION_CONTRACT_VERSION:
        raise ValueError(
            f"Feature-AE model {model_version!r} has thresholds from {thresholds.get('score_contract_version')!r}, "
            f"expected {FEATURE_AE_CHAMPION_CONTRACT_VERSION!r}"
        )


def resolve_model_checkpoint(
    model_version: str,
    *,
    cache_root: str | Path | None = None,
    strict_checksum: bool = False,
    s3_client: Any | None = None,
) -> Path:
    return resolve_model_artifact_from_manifest(
        model_manifest_path(model_version),
        cache_root=cache_root,
        strict_checksum=strict_checksum,
        s3_client=s3_client,
    )


def resolve_roi_segmenter_checkpoint(
    version: str = DEFAULT_ROI_MODEL_VERSION,
    *,
    cache_root: str | Path | None = None,
    strict_checksum: bool = False,
    s3_client: Any | None = None,
) -> Path:
    return resolve_model_checkpoint(
        version,
        cache_root=cache_root,
        strict_checksum=strict_checksum,
        s3_client=s3_client,
    )


def resolve_feature_ae_checkpoint(
    version: str = DEFAULT_FEATURE_AE_MODEL_VERSION,
    *,
    cache_root: str | Path | None = None,
    strict_checksum: bool = False,
    s3_client: Any | None = None,
) -> Path:
    return resolve_model_checkpoint(
        version,
        cache_root=cache_root,
        strict_checksum=strict_checksum,
        s3_client=s3_client,
    )


__all__ = [
    "DEFAULT_FEATURE_AE_MODEL_VERSION",
    "DEFAULT_ROI_MODEL_VERSION",
    "load_feature_ae_decision_thresholds",
    "load_model_manifest",
    "model_manifest_path",
    "resolve_feature_ae_checkpoint",
    "resolve_model_checkpoint",
    "resolve_roi_segmenter_checkpoint",
    "validate_feature_ae_champion_manifest",
]
