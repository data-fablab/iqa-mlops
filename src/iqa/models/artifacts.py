"""Model-version helpers resolving checkpoint manifests to local cache paths."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from iqa.models.feature_ae.reference import FeatureAEReferenceContract

from iqa.storage.artifacts import resolve_model_artifact_from_manifest

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_MANIFESTS_DIR = DEFAULT_REPO_ROOT / "models" / "manifests"
DEFAULT_ROI_MODEL_VERSION = "roi_segmenter_v001_fixed"
DEFAULT_FEATURE_AE_MODEL_VERSION = "rd_feature_ae_gated_v001_bootstrap"
FEATURE_AE_REFERENCE_CONTRACT_VERSION = "feature_ae_reference_v001"
FEATURE_AE_REFERENCE_REQUIRED_FIELDS = {
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
) -> dict[str, Any]:
    manifest = load_model_manifest(model_version)
    validate_feature_ae_reference_manifest(manifest, model_version=model_version)
    thresholds = manifest.get("decision_thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError(f"Feature-AE model {model_version!r} is missing calibrated decision_thresholds")
    return thresholds


def load_feature_ae_reference_contract(
    model_version: str = DEFAULT_FEATURE_AE_MODEL_VERSION,
) -> FeatureAEReferenceContract:
    manifest = load_model_manifest(model_version)
    validate_feature_ae_reference_manifest(manifest, model_version=model_version)
    return feature_ae_reference_contract_from_payload(manifest["feature_ae_reference_contract"])


def feature_ae_reference_contract_from_payload(payload: dict[str, Any]) -> FeatureAEReferenceContract:
    from iqa.models.feature_ae.reference import (
        REFERENCE_FEATURE_AE_CONTRACT,
        FeatureAEReferenceContract,
    )

    version = payload.get("version") or payload.get("score_contract_version")
    layer_weights = payload.get("layer_weights")
    normalization_stats = payload.get("layer_normalization_stats")

    return FeatureAEReferenceContract(
        version=str(version or REFERENCE_FEATURE_AE_CONTRACT.version),
        teacher_weights=str(payload.get("teacher_weights", REFERENCE_FEATURE_AE_CONTRACT.teacher_weights)),
        tile_size=int(payload.get("tile_size", REFERENCE_FEATURE_AE_CONTRACT.tile_size)),
        context_size=int(payload.get("context_size", REFERENCE_FEATURE_AE_CONTRACT.context_size)),
        tile_stride=int(payload.get("tile_stride", REFERENCE_FEATURE_AE_CONTRACT.tile_stride)),
        layers=tuple(str(layer) for layer in payload.get("layers", REFERENCE_FEATURE_AE_CONTRACT.layers)),
        layer_weights=(
            {str(name): float(value) for name, value in layer_weights.items()}
            if isinstance(layer_weights, dict)
            else REFERENCE_FEATURE_AE_CONTRACT.normalized_layer_weights()
        ),
        score_smoothing=str(payload.get("score_smoothing", REFERENCE_FEATURE_AE_CONTRACT.score_smoothing)),
        roi_mode=str(payload.get("roi_mode", REFERENCE_FEATURE_AE_CONTRACT.roi_mode)),
        roi_threshold=float(payload.get("roi_threshold", REFERENCE_FEATURE_AE_CONTRACT.roi_threshold)),
        score_image=str(payload.get("score_image", REFERENCE_FEATURE_AE_CONTRACT.score_image)),
        topk_fraction=float(payload.get("topk_fraction", REFERENCE_FEATURE_AE_CONTRACT.topk_fraction)),
        layer_score_mode=str(
            payload.get("layer_score_mode", REFERENCE_FEATURE_AE_CONTRACT.layer_score_mode)
        ),
        layer_normalization=str(
            payload.get("layer_normalization", REFERENCE_FEATURE_AE_CONTRACT.layer_normalization)
        ),
        layer_normalization_stats=(
            {str(name): float(value) for name, value in normalization_stats.items()}
            if isinstance(normalization_stats, dict)
            else None
        ),
        cosine_weight=float(
            payload.get("cosine_weight", REFERENCE_FEATURE_AE_CONTRACT.cosine_weight)
        ),
    )


def load_feature_ae_runtime_contract(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    reference_contract = payload.get("feature_ae_reference_contract") or payload.get("score_contract")
    if not isinstance(reference_contract, dict):
        raise ValueError(f"Feature-AE runtime contract is missing feature_ae_reference_contract: {path}")
    thresholds = payload.get("decision_thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError(f"Feature-AE runtime contract is missing decision_thresholds: {path}")
    contract_version = reference_contract.get("version") or reference_contract.get("score_contract_version")
    if contract_version != FEATURE_AE_REFERENCE_CONTRACT_VERSION:
        raise ValueError(
            f"Feature-AE runtime contract uses unsupported score contract {contract_version!r}"
        )
    if thresholds.get("score_contract_version") != FEATURE_AE_REFERENCE_CONTRACT_VERSION:
        raise ValueError(
            f"Feature-AE runtime thresholds use unsupported score contract "
            f"{thresholds.get('score_contract_version')!r}"
        )
    return payload


def validate_feature_ae_reference_manifest(
    manifest: dict[str, Any],
    *,
    model_version: str = DEFAULT_FEATURE_AE_MODEL_VERSION,
) -> None:
    contract = manifest.get("feature_ae_reference_contract")
    if not isinstance(contract, dict):
        raise ValueError(f"Feature-AE model {model_version!r} is missing feature_ae_reference_contract")
    missing = sorted(FEATURE_AE_REFERENCE_REQUIRED_FIELDS - set(contract))
    if missing:
        raise ValueError(
            f"Feature-AE model {model_version!r} has incomplete reference contract: missing {', '.join(missing)}"
        )
    if contract.get("version") != FEATURE_AE_REFERENCE_CONTRACT_VERSION:
        raise ValueError(
            f"Feature-AE model {model_version!r} uses unsupported score contract {contract.get('version')!r}"
        )
    thresholds = manifest.get("decision_thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError(f"Feature-AE model {model_version!r} is missing calibrated decision_thresholds")
    if isinstance(thresholds, dict) and thresholds.get("score_contract_version") != FEATURE_AE_REFERENCE_CONTRACT_VERSION:
        raise ValueError(
            f"Feature-AE model {model_version!r} has thresholds from {thresholds.get('score_contract_version')!r}, "
            f"expected {FEATURE_AE_REFERENCE_CONTRACT_VERSION!r}"
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
    "load_feature_ae_reference_contract",
    "load_feature_ae_runtime_contract",
    "load_model_manifest",
    "model_manifest_path",
    "feature_ae_reference_contract_from_payload",
    "resolve_feature_ae_checkpoint",
    "resolve_model_checkpoint",
    "resolve_roi_segmenter_checkpoint",
    "validate_feature_ae_reference_manifest",
]
