from __future__ import annotations

import json
from pathlib import Path

import pytest

from iqa.models import artifacts as model_artifacts
from iqa.storage.artifacts import (
    PENDING_CHECKSUM,
    load_model_artifact_manifest,
    resolve_model_artifact_from_manifest,
    resolve_model_artifact_uri,
    sha256_file,
)


class FakeS3Client:
    def __init__(self, source: Path):
        self.source = source
        self.calls: list[tuple[str, str, str]] = []

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        self.calls.append((bucket, key, filename))
        Path(filename).write_bytes(self.source.read_bytes())


def _write_manifest(path: Path, *, artifact_uri: str, sha256: str | None = None, model_version: str = "demo") -> None:
    payload = {"model_version": model_version, "artifact_uri": artifact_uri}
    if sha256 is not None:
        payload["sha256"] = sha256
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_model_artifact_manifest_requires_artifact_uri(tmp_path: Path) -> None:
    manifest = tmp_path / "model_manifest.json"
    manifest.write_text(json.dumps({"model_version": "demo"}), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact_uri"):
        load_model_artifact_manifest(manifest)


def test_resolves_local_artifact_and_verifies_checksum(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    manifest = tmp_path / "model_manifest.json"
    _write_manifest(manifest, artifact_uri=str(checkpoint), sha256=sha256_file(checkpoint))

    resolved = resolve_model_artifact_from_manifest(manifest)

    assert resolved == checkpoint


def test_checksum_mismatch_raises(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="Checksum mismatch"):
        resolve_model_artifact_uri(str(checkpoint), model_version="demo", sha256="0" * 64)


def test_strict_checksum_rejects_pending_restore(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="Strict checksum"):
        resolve_model_artifact_uri(
            str(checkpoint),
            model_version="demo",
            sha256=PENDING_CHECKSUM,
            strict_checksum=True,
        )


def test_downloads_s3_artifact_to_cache_and_reuses_it(tmp_path: Path) -> None:
    source = tmp_path / "remote.pt"
    source.write_bytes(b"remote-checkpoint")
    client = FakeS3Client(source)
    cache_root = tmp_path / "cache"

    resolved = resolve_model_artifact_uri(
        "s3://iqa-models/demo/checkpoint.pt",
        model_version="demo",
        sha256=sha256_file(source),
        cache_root=cache_root,
        s3_client=client,
    )
    resolved_again = resolve_model_artifact_uri(
        "s3://iqa-models/demo/checkpoint.pt",
        model_version="demo",
        sha256=sha256_file(source),
        cache_root=cache_root,
        s3_client=client,
    )

    assert resolved == cache_root / "demo" / "checkpoint.pt"
    assert resolved.read_bytes() == b"remote-checkpoint"
    assert resolved_again == resolved
    assert len(client.calls) == 1
    assert client.calls[0][0:2] == ("iqa-models", "demo/checkpoint.pt")


def test_s3_directory_uri_appends_checkpoint_filename(tmp_path: Path) -> None:
    source = tmp_path / "checkpoint.pt"
    source.write_bytes(b"remote-checkpoint")
    client = FakeS3Client(source)

    resolve_model_artifact_uri(
        "s3://mlflow-artifacts/run123/artifacts/model",
        model_version="demo",
        cache_root=tmp_path / "cache",
        s3_client=client,
    )

    assert client.calls[0][0:2] == ("mlflow-artifacts", "run123/artifacts/model/checkpoint.pt")


def test_model_helpers_resolve_manifests_with_mocked_s3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "checkpoint.pt"
    source.write_bytes(b"remote-checkpoint")
    client = FakeS3Client(source)
    manifests_dir = tmp_path / "manifests"
    roi_dir = manifests_dir / model_artifacts.DEFAULT_ROI_MODEL_VERSION
    feature_dir = manifests_dir / model_artifacts.DEFAULT_FEATURE_AE_MODEL_VERSION
    roi_dir.mkdir(parents=True)
    feature_dir.mkdir(parents=True)
    _write_manifest(
        roi_dir / "model_manifest.json",
        artifact_uri="s3://iqa-models/roi/checkpoint.pt",
        model_version=model_artifacts.DEFAULT_ROI_MODEL_VERSION,
    )
    _write_manifest(
        feature_dir / "model_manifest.json",
        artifact_uri="s3://iqa-models/feature/checkpoint.pt",
        model_version=model_artifacts.DEFAULT_FEATURE_AE_MODEL_VERSION,
    )
    monkeypatch.setattr(model_artifacts, "MODEL_MANIFESTS_DIR", manifests_dir)

    roi = model_artifacts.resolve_roi_segmenter_checkpoint(cache_root=tmp_path / "cache", s3_client=client)
    feature = model_artifacts.resolve_feature_ae_checkpoint(cache_root=tmp_path / "cache", s3_client=client)

    assert roi.name == "checkpoint.pt"
    assert feature.name == "checkpoint.pt"


def test_model_manifest_path_honours_repo_root_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Airflow task containers run installed code from /app but mount the repo elsewhere."""
    repo_root = tmp_path / "mounted-repo"
    monkeypatch.setenv("IQA_REPO_ROOT", str(repo_root))

    path = model_artifacts.model_manifest_path("demo_model")

    assert path == repo_root / "models" / "manifests" / "demo_model" / "model_manifest.json"


def test_load_feature_ae_reference_contract_from_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifests_dir = tmp_path / "manifests"
    model_dir = manifests_dir / model_artifacts.DEFAULT_FEATURE_AE_MODEL_VERSION
    model_dir.mkdir(parents=True)

    manifest = {
        "model_version": model_artifacts.DEFAULT_FEATURE_AE_MODEL_VERSION,
        "artifact_uri": "s3://iqa-models/feature/checkpoint.pt",
        "decision_thresholds": {
            "score_contract_version": model_artifacts.FEATURE_AE_REFERENCE_CONTRACT_VERSION,
            "threshold_orange": 12.0,
            "threshold_red": 20.0,
        },
        "feature_ae_reference_contract": {
            "version": model_artifacts.FEATURE_AE_REFERENCE_CONTRACT_VERSION,
            "teacher_weights": "IMAGENET1K_V1",
            "tile_size": 384,
            "context_size": 768,
            "tile_stride": 384,
            "layers": ["layer2", "layer3"],
            "layer_weights": {"layer2": 0.65, "layer3": 0.35},
            "score_smoothing": "median3",
            "roi_mode": "soft_map",
            "roi_threshold": 0.5,
            "score_image": "topk_mean",
            "topk_fraction": 0.005,
            "layer_score_mode": "sqrt_l2_plus_cosine",
            "layer_normalization": "good_p99",
            "layer_normalization_stats": {"layer2": 2.0, "layer3": 4.0},
            "cosine_weight": 0.75,
        },
    }
    (model_dir / "model_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    monkeypatch.setattr(model_artifacts, "MODEL_MANIFESTS_DIR", manifests_dir)

    contract = model_artifacts.load_feature_ae_reference_contract()

    assert contract.layers == ("layer2", "layer3")
    assert contract.layer_weights == {"layer2": 0.65, "layer3": 0.35}
    assert contract.layer_normalization_stats == {"layer2": 2.0, "layer3": 4.0}
    assert contract.roi_mode == "soft_map"
    assert contract.topk_fraction == pytest.approx(0.005)
    assert contract.cosine_weight == pytest.approx(0.75)
