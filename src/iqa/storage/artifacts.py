"""Model artifact resolution from local paths or MinIO/S3 URIs."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from iqa.storage.uris import parse_s3_uri

PENDING_CHECKSUM = "pending-restore"
DEFAULT_MODEL_CACHE_DIR = Path(".cache") / "iqa" / "models"


@dataclass(frozen=True)
class ModelArtifactManifest:
    model_version: str
    artifact_uri: str
    sha256: str | None = None


def load_model_artifact_manifest(path: str | Path) -> ModelArtifactManifest:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_version = str(payload.get("model_version") or "").strip()
    artifact_uri = str(payload.get("artifact_uri") or "").strip()
    if not model_version:
        raise ValueError(f"Model manifest {manifest_path} is missing model_version.")
    if not artifact_uri:
        raise ValueError(f"Model manifest {manifest_path} is missing artifact_uri.")
    sha256 = payload.get("sha256")
    return ModelArtifactManifest(
        model_version=model_version,
        artifact_uri=artifact_uri,
        sha256=str(sha256).strip() if sha256 is not None else None,
    )


def resolve_model_artifact_from_manifest(
    manifest_path: str | Path,
    *,
    cache_root: str | Path | None = None,
    strict_checksum: bool = False,
    s3_client: Any | None = None,
    filename: str = "checkpoint.pt",
) -> Path:
    manifest = load_model_artifact_manifest(manifest_path)
    return resolve_model_artifact_uri(
        manifest.artifact_uri,
        model_version=manifest.model_version,
        sha256=manifest.sha256,
        cache_root=cache_root,
        strict_checksum=strict_checksum,
        s3_client=s3_client,
        filename=filename,
    )


def resolve_model_artifact_uri(
    artifact_uri: str,
    *,
    model_version: str,
    sha256: str | None = None,
    cache_root: str | Path | None = None,
    strict_checksum: bool = False,
    s3_client: Any | None = None,
    filename: str = "checkpoint.pt",
) -> Path:
    _validate_checksum_policy(sha256, strict_checksum=strict_checksum)
    if artifact_uri.startswith("s3://"):
        return _resolve_s3_artifact(
            artifact_uri,
            model_version=model_version,
            sha256=sha256,
            cache_root=_cache_root(cache_root),
            s3_client=s3_client,
            filename=filename,
        )
    return _resolve_local_artifact(artifact_uri, sha256=sha256, filename=filename)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checksum_is_verifiable(sha256: str | None) -> bool:
    return bool(sha256 and sha256 != PENDING_CHECKSUM)


def _resolve_local_artifact(artifact_uri: str, *, sha256: str | None, filename: str) -> Path:
    path = Path(artifact_uri.removeprefix("file://"))
    if path.is_dir():
        path = path / filename
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    _verify_checksum(path, sha256)
    return path


def _resolve_s3_artifact(
    artifact_uri: str,
    *,
    model_version: str,
    sha256: str | None,
    cache_root: Path,
    s3_client: Any | None,
    filename: str,
) -> Path:
    parsed = parse_s3_uri(_checkpoint_s3_uri(artifact_uri, filename=filename))
    cache_path = cache_root / model_version / Path(parsed.key).name
    if cache_path.exists():
        _verify_checksum(cache_path, sha256)
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    client = s3_client or _build_s3_client()
    client.download_file(parsed.bucket, parsed.key, str(cache_path))
    _verify_checksum(cache_path, sha256)
    return cache_path


def _checkpoint_s3_uri(uri: str, *, filename: str) -> str:
    if uri.endswith((".pt", ".pth", ".ckpt", ".onnx")):
        return uri
    return f"{uri.rstrip('/')}/{filename}"


def _cache_root(cache_root: str | Path | None) -> Path:
    if cache_root is not None:
        return Path(cache_root)
    configured = os.environ.get("IQA_MODEL_CACHE_DIR")
    return Path(configured) if configured else DEFAULT_MODEL_CACHE_DIR


def _verify_checksum(path: Path, sha256: str | None) -> None:
    if not checksum_is_verifiable(sha256):
        return
    actual = sha256_file(path)
    if actual != sha256:
        raise ValueError(f"Checksum mismatch for {path}: expected {sha256}, got {actual}.")


def _validate_checksum_policy(sha256: str | None, *, strict_checksum: bool) -> None:
    if strict_checksum and not checksum_is_verifiable(sha256):
        raise ValueError("Strict checksum verification requested, but model manifest has no verifiable sha256.")


def _build_s3_client() -> Any:
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - dependency is present in normal envs.
        raise ImportError("boto3 is required to download model artifacts from S3/MinIO.") from exc

    kwargs: dict[str, str] = {}
    if endpoint_url := os.environ.get("IQA_S3_ENDPOINT_URL"):
        kwargs["endpoint_url"] = endpoint_url
    if access_key := os.environ.get("IQA_S3_ACCESS_KEY_ID"):
        kwargs["aws_access_key_id"] = access_key
    if secret_key := os.environ.get("IQA_S3_SECRET_ACCESS_KEY"):
        kwargs["aws_secret_access_key"] = secret_key
    if region := os.environ.get("IQA_S3_REGION"):
        kwargs["region_name"] = region
    return boto3.client("s3", **kwargs)


__all__ = [
    "DEFAULT_MODEL_CACHE_DIR",
    "ModelArtifactManifest",
    "PENDING_CHECKSUM",
    "checksum_is_verifiable",
    "load_model_artifact_manifest",
    "resolve_model_artifact_from_manifest",
    "resolve_model_artifact_uri",
    "sha256_file",
]
