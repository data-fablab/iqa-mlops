"""Helpers for publishing visual runtime artifacts to MinIO/S3."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from iqa.storage.object_store import InMemoryObjectStore, ObjectStore, S3ObjectStore, create_object_store
from iqa.storage.uris import IQA_BUCKETS


@dataclass(frozen=True)
class VisualArtifactContext:
    scenario_id: str
    lot_id: str
    piece_event_id: str
    image_id: str


def create_visual_object_store() -> ObjectStore:
    """Create an object store for runtime visual artifacts.

    The general object-store factory defaults to memory for offline safety. Visual
    artifacts are a runtime/server feature, so the presence of an S3 endpoint is
    enough to opt into MinIO unless a backend was set explicitly.
    """

    if os.getenv("IQA_OBJECT_STORE_BACKEND"):
        return create_object_store()
    if os.getenv("IQA_S3_ENDPOINT_URL"):
        return S3ObjectStore()
    return InMemoryObjectStore()


def visual_artifact_key(context: VisualArtifactContext, *, artifact: str, suffix: str = "png") -> str:
    scenario = _safe_segment(context.scenario_id or "unknown_scenario")
    lot = _safe_segment(context.lot_id or "unknown_lot")
    piece = _safe_segment(context.piece_event_id or "unknown_piece")
    image = _safe_segment(context.image_id or "unknown_image")
    artifact_name = _safe_segment(artifact)
    return f"lots/{scenario}/{lot}/{piece}_{image}_{artifact_name}.{suffix.lstrip('.')}"


def publish_visual_artifact(
    path: str | Path,
    *,
    bucket: str,
    key: str,
    store: ObjectStore | None = None,
    content_type: str = "image/png",
) -> str:
    payload = Path(path).read_bytes()
    return (store or create_visual_object_store()).put_bytes(bucket, key, payload, content_type=content_type)


def publish_roi_mask(
    path: str | Path,
    context: VisualArtifactContext,
    *,
    store: ObjectStore | None = None,
) -> str:
    return publish_visual_artifact(
        path,
        bucket=IQA_BUCKETS["roi_masks"],
        key=visual_artifact_key(context, artifact="roi"),
        store=store,
    )


def publish_heatmap(
    path: str | Path,
    context: VisualArtifactContext,
    *,
    store: ObjectStore | None = None,
) -> str:
    return publish_visual_artifact(
        path,
        bucket=IQA_BUCKETS["heatmaps"],
        key=visual_artifact_key(context, artifact="heatmap"),
        store=store,
    )


def _safe_segment(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return cleaned.strip("._") or "unknown"


__all__ = [
    "VisualArtifactContext",
    "create_visual_object_store",
    "publish_heatmap",
    "publish_roi_mask",
    "publish_visual_artifact",
    "visual_artifact_key",
]
