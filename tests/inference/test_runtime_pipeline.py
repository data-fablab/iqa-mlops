"""Tests for the real ROI then Feature AE runtime."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from PIL import Image

from iqa.inference.contracts import InferenceRequest
from iqa.inference.feature_ae import FeatureAEPrediction
from iqa.inference.segmentation import RoiSegmentationPrediction
from iqa.models.feature_ae.reference import REFERENCE_FEATURE_AE_CONTRACT
from iqa.storage.object_store import InMemoryObjectStore
from iqa.storage.uris import IQA_BUCKETS, parse_s3_uri


def _image_bytes(path: Path) -> bytes:
    Image.new("RGB", (16, 16), color=(120, 120, 120)).save(path)
    return path.read_bytes()


def _patch_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from iqa.inference import runtime

    roi_checkpoint = tmp_path / "roi.pt"
    feature_checkpoint = tmp_path / "feature.pt"
    roi_checkpoint.write_bytes(b"roi")
    feature_checkpoint.write_bytes(b"feature")

    monkeypatch.setattr(
        runtime,
        "resolve_roi_segmenter_checkpoint",
        lambda **kwargs: roi_checkpoint,
    )
    monkeypatch.setattr(
        runtime,
        "resolve_feature_ae_checkpoint",
        lambda **kwargs: feature_checkpoint,
    )
    monkeypatch.setattr(
        runtime,
        "load_feature_ae_decision_thresholds",
        lambda model_version: {
            "threshold_orange": 100.0,
            "threshold_red": 200.0,
        },
    )
    monkeypatch.setattr(
        runtime,
        "load_feature_ae_reference_contract",
        lambda model_version: REFERENCE_FEATURE_AE_CONTRACT,
    )

    def fake_roi(
        image_path: str | Path,
        checkpoint_path: str | Path,
        **kwargs,
    ) -> RoiSegmentationPrediction:
        Image.new("L", (16, 16), color=255).save(kwargs["output_mask"])
        Image.new("L", (16, 16), color=200).save(kwargs["output_probability_map"])
        return RoiSegmentationPrediction(
            model_type="roi_test",
            checkpoint_path=str(checkpoint_path),
            image_path=str(image_path),
            roi_ratio=0.75,
            roi_quality_status="ok",
            threshold=0.5,
            image_size=384,
            context_size=384,
            mask_mode="argmax",
        )

    def fake_feature(
        image_path: str | Path,
        checkpoint_path: str | Path,
        **kwargs,
    ) -> FeatureAEPrediction:
        Image.new("RGB", (16, 16), color=(255, 100, 0)).save(kwargs["heatmap_output_path"])
        return FeatureAEPrediction(
            image_path=str(image_path),
            model_type="feature_test",
            score=150.0,
            status="orange",
            threshold_orange=100.0,
            threshold_red=200.0,
            latency_ms=10.0,
            roi_status="roi_scored",
            heatmap_uri=None,
            threshold_source="manifest",
            score_contract_version=REFERENCE_FEATURE_AE_CONTRACT.version,
        )

    monkeypatch.setattr(runtime, "predict_roi_image", fake_roi)
    monkeypatch.setattr(runtime, "predict_feature_ae_image", fake_feature)


def test_run_inference_pipeline_from_local_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from iqa.inference.runtime import run_inference_pipeline

    _patch_runtime(monkeypatch, tmp_path)
    image_path = tmp_path / "piece.jpg"
    payload = _image_bytes(image_path)
    store = InMemoryObjectStore()

    pipeline = run_inference_pipeline(
        InferenceRequest(
            piece_event_id="piece_001",
            scenario_id="production_replay_natural",
            image_uri=str(image_path),
            sha256=hashlib.sha256(payload).hexdigest(),
            lot_id="LOT_001",
            dataset_version="casting_v001",
        ),
        store=store,
        device="cpu",
    )

    assert pipeline.result.score == 150.0
    assert pipeline.result.decision == "Orange"
    assert pipeline.result.roi_status == "ok"
    assert pipeline.result.roi_model_version == "roi_segmenter_v001_fixed"
    assert pipeline.result.feature_ae_version == "rd_feature_ae_gated_v001_bootstrap"
    assert parse_s3_uri(pipeline.result.heatmap_uri).bucket == IQA_BUCKETS["heatmaps"]
    assert pipeline.roi_prediction is not None
    assert parse_s3_uri(pipeline.roi_prediction.roi_mask_uri).bucket == IQA_BUCKETS["roi_masks"]


def test_run_inference_pipeline_from_s3_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from iqa.inference.runtime import run_inference_pipeline

    _patch_runtime(monkeypatch, tmp_path)
    source_path = tmp_path / "source.jpg"
    payload = _image_bytes(source_path)
    store = InMemoryObjectStore(
        {(IQA_BUCKETS["ingested_images"], "incoming/piece.jpg"): payload}
    )

    pipeline = run_inference_pipeline(
        InferenceRequest(
            piece_event_id="piece_002",
            scenario_id="production_replay_natural",
            image_uri="s3://iqa-ingested-images/incoming/piece.jpg",
            lot_id="LOT_002",
        ),
        store=store,
        device="cpu",
    )

    heatmap = parse_s3_uri(pipeline.result.heatmap_uri)
    roi_mask = parse_s3_uri(pipeline.roi_prediction.roi_mask_uri)

    assert store.exists(heatmap.bucket, heatmap.key)
    assert store.exists(roi_mask.bucket, roi_mask.key)


def test_run_inference_pipeline_rejects_missing_local_image(
    tmp_path: Path,
) -> None:
    from iqa.inference.runtime import run_inference_pipeline

    with pytest.raises(FileNotFoundError, match="Input image not found"):
        run_inference_pipeline(
            InferenceRequest(
                piece_event_id="piece_missing",
                scenario_id="production_replay_natural",
                image_uri=str(tmp_path / "missing.jpg"),
            ),
            store=InMemoryObjectStore(),
            device="cpu",
        )


def test_run_inference_pipeline_rejects_sha256_mismatch(
    tmp_path: Path,
) -> None:
    from iqa.inference.runtime import run_inference_pipeline

    image_path = tmp_path / "piece.jpg"
    _image_bytes(image_path)

    with pytest.raises(ValueError, match="Input image checksum mismatch"):
        run_inference_pipeline(
            InferenceRequest(
                piece_event_id="piece_bad_sha",
                scenario_id="production_replay_natural",
                image_uri=str(image_path),
                sha256="0" * 64,
            ),
            store=InMemoryObjectStore(),
            device="cpu",
        )
