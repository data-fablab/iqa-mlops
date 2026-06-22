"""Feature-AE bootstrap checkpoint selection and publication helpers."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from iqa.storage.artifacts import sha256_file
from iqa.storage.uris import parse_s3_uri
from iqa.training.feature_ae_contracts import (
    FEATURE_AE_BUSINESS_METRIC_PRIORITY,
    FEATURE_AE_PREPROCESSING_CONTRACT_VERSION,
    canonical_feature_ae_preprocessing_dict,
)

BOOTSTRAP_MODEL_VERSION = "rd_feature_ae_gated_v001_bootstrap"
BOOTSTRAP_ARTIFACT_URI = f"s3://iqa-models/{BOOTSTRAP_MODEL_VERSION}/checkpoint.pt"
BOOTSTRAP_RUNTIME_CACHE_CHECKPOINT = Path(".cache/iqa/models") / BOOTSTRAP_MODEL_VERSION / "checkpoint.pt"
BUSINESS_METRIC_PRIORITY = FEATURE_AE_BUSINESS_METRIC_PRIORITY


@dataclass(frozen=True)
class BootstrapChampion:
    checkpoint_path: Path
    checkpoint_name: str
    selected_metric: str
    selected_metric_value: float
    selected_epoch: int
    sha256: str
    val_loss: float | None = None

    def to_manifest_fields(self) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "sha256": self.sha256,
            "selected_metric": self.selected_metric,
            "selected_metric_value": self.selected_metric_value,
            "selected_epoch": self.selected_epoch,
            "source_checkpoint": self.checkpoint_name,
        }
        if self.val_loss is not None:
            fields["selected_val_loss"] = self.val_loss
        return fields


def select_bootstrap_champion(run_dir: str | Path) -> BootstrapChampion:
    """Select a bootstrap checkpoint by business metrics, never by loss first."""
    path = Path(run_dir)
    best_path = path / "metric_eval_best.json"
    if not best_path.is_file():
        raise FileNotFoundError(f"Feature-AE bootstrap selection requires {best_path}.")
    best = json.loads(best_path.read_text(encoding="utf-8"))
    losses = _read_val_losses(path / "loss_history.csv")

    for metric in BUSINESS_METRIC_PRIORITY:
        record = best.get(metric)
        if not record:
            continue
        value = record.get("value")
        if value is None or not math.isfinite(float(value)):
            continue
        checkpoint_name = str(record.get("checkpoint") or "").strip()
        if not checkpoint_name:
            raise ValueError(f"Metric {metric} has no checkpoint in {best_path}.")
        checkpoint_path = path / checkpoint_name
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Selected checkpoint for {metric} is missing: {checkpoint_path}")
        epoch = int(record.get("epoch") or 0)
        return BootstrapChampion(
            checkpoint_path=checkpoint_path,
            checkpoint_name=checkpoint_name,
            selected_metric=metric,
            selected_metric_value=float(value),
            selected_epoch=epoch,
            sha256=sha256_file(checkpoint_path),
            val_loss=losses.get(epoch),
        )

    raise ValueError(
        "No bootstrap business metric is available. Expected one of: "
        + ", ".join(BUSINESS_METRIC_PRIORITY)
    )


def materialize_bootstrap_checkpoint(champion: BootstrapChampion, output_checkpoint: str | Path) -> Path:
    """Copy the selected champion to the canonical checkpoint path."""
    destination = Path(output_checkpoint)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(champion.checkpoint_path, destination)
    copied_sha256 = sha256_file(destination)
    if copied_sha256 != champion.sha256:
        raise ValueError(f"Copied bootstrap checkpoint checksum mismatch: expected {champion.sha256}, got {copied_sha256}.")
    return destination


def sync_bootstrap_runtime_cache(
    source_checkpoint: str | Path,
    runtime_checkpoint: str | Path = BOOTSTRAP_RUNTIME_CACHE_CHECKPOINT,
) -> Path:
    """Keep the model resolver cache aligned with the freshly selected bootstrap champion."""
    source = Path(source_checkpoint)
    destination = Path(runtime_checkpoint)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_sha256 = sha256_file(source)
    copied_sha256 = sha256_file(destination)
    if copied_sha256 != source_sha256:
        raise ValueError(
            f"Runtime bootstrap checkpoint checksum mismatch: expected {source_sha256}, got {copied_sha256}."
        )
    return destination


def update_bootstrap_manifest(
    manifest_path: str | Path,
    champion: BootstrapChampion,
    *,
    artifact_uri: str = BOOTSTRAP_ARTIFACT_URI,
    dataset_version: str = "feature_ae_good_v001_bootstrap",
    validation_set_id: str = "validation_set_v001",
    roi_model_version: str = "roi_segmenter_v001_fixed",
) -> dict[str, Any]:
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(
        {
            "model_version": BOOTSTRAP_MODEL_VERSION,
            "status": "bootstrap",
            "artifact_uri": artifact_uri,
            "dataset_version": dataset_version,
            "validation_set_id": validation_set_id,
            "roi_model_version": roi_model_version,
            "preprocessing_contract": canonical_feature_ae_preprocessing_dict(),
            "preprocessing_contract_version": FEATURE_AE_PREPROCESSING_CONTRACT_VERSION,
            "checkpoint_selection_policy": "business_metric_only",
            "selection_policy": "pixel_aupimo_1e-5_1e-3 -> pixel_ap -> image_ap -> image_auroc; val_loss informational only",
        }
    )
    payload.update(champion.to_manifest_fields())
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def upload_checkpoint_to_s3(checkpoint_path: str | Path, artifact_uri: str = BOOTSTRAP_ARTIFACT_URI, *, s3_client: Any | None = None) -> None:
    parsed = parse_s3_uri(artifact_uri)
    client = s3_client or _build_s3_client()
    client.upload_file(str(checkpoint_path), parsed.bucket, parsed.key)


def _read_val_losses(path: Path) -> dict[int, float]:
    if not path.is_file():
        return {}
    losses: dict[int, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                losses[int(row["epoch"])] = float(row["val_loss"])
            except (KeyError, TypeError, ValueError):
                continue
    return losses


def _build_s3_client() -> Any:
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - dependency is present in normal envs.
        raise ImportError("boto3 is required to upload Feature-AE bootstrap artifacts to S3/MinIO.") from exc

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
    "BOOTSTRAP_ARTIFACT_URI",
    "BOOTSTRAP_MODEL_VERSION",
    "BOOTSTRAP_RUNTIME_CACHE_CHECKPOINT",
    "BUSINESS_METRIC_PRIORITY",
    "BootstrapChampion",
    "materialize_bootstrap_checkpoint",
    "select_bootstrap_champion",
    "sync_bootstrap_runtime_cache",
    "update_bootstrap_manifest",
    "upload_checkpoint_to_s3",
]
