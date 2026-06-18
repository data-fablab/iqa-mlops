"""Model-version helpers resolving checkpoint manifests to local cache paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from iqa.storage.artifacts import resolve_model_artifact_from_manifest

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_MANIFESTS_DIR = REPO_ROOT / "models" / "manifests"
DEFAULT_ROI_MODEL_VERSION = "roi_segmenter_v001_fixed"
DEFAULT_FEATURE_AE_MODEL_VERSION = "rd_feature_ae_gated_v001_bootstrap"


def model_manifest_path(model_version: str) -> Path:
    return MODEL_MANIFESTS_DIR / model_version / "model_manifest.json"


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
    "model_manifest_path",
    "resolve_feature_ae_checkpoint",
    "resolve_model_checkpoint",
    "resolve_roi_segmenter_checkpoint",
]
