"""Real IQA runtime orchestrating ROI segmentation then Feature AE."""

from __future__ import annotations

import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

from iqa.inference.contracts import InferenceRequest, InferenceResult
from iqa.inference.feature_ae import predict_feature_ae_image
from iqa.inference.pipeline import InferencePipelineResult, decision_from_roi_and_score
from iqa.inference.segmentation import predict_roi_image
from iqa.models.artifacts import (
    DEFAULT_FEATURE_AE_MODEL_VERSION,
    DEFAULT_ROI_MODEL_VERSION,
    load_feature_ae_decision_thresholds,
    load_feature_ae_reference_contract,
    resolve_feature_ae_checkpoint,
    resolve_roi_segmenter_checkpoint,
)
from iqa.roi.artifacts import RoiPredictionArtifact
from iqa.storage.artifacts import sha256_file
from iqa.storage.object_store import ObjectStore, create_object_store
from iqa.storage.uris import parse_s3_uri
from iqa.storage.visual_artifacts import (
    VisualArtifactContext,
    publish_heatmap,
    publish_roi_mask,
)


def _local_image_path(image_uri: str) -> Path:
    if image_uri.startswith("file://"):
        parsed = urlparse(image_uri)
        if parsed.netloc not in {"", "localhost"}:
            raise ValueError(f"Unsupported file URI host: {parsed.netloc!r}")
        return Path(unquote(parsed.path))
    return Path(image_uri)


def _image_suffix(image_uri: str) -> str:
    if image_uri.startswith("s3://"):
        suffix = Path(parse_s3_uri(image_uri).key).suffix
    else:
        suffix = _local_image_path(image_uri).suffix
    return suffix or ".img"


def _image_id(image_uri: str) -> str:
    if image_uri.startswith("s3://"):
        return Path(parse_s3_uri(image_uri).key).stem or "image"
    return _local_image_path(image_uri).stem or "image"


def _resolve_input_image(
    image_uri: str,
    *,
    work_dir: Path,
    store: ObjectStore,
) -> Path:
    if image_uri.startswith("s3://"):
        parsed = parse_s3_uri(image_uri)
        try:
            payload = store.get_bytes(parsed.bucket, parsed.key)
        except KeyError as error:
            raise FileNotFoundError(f"Input image not found: {image_uri}") from error
        image_path = work_dir / f"input{_image_suffix(image_uri)}"
        image_path.write_bytes(payload)
        return image_path

    image_path = _local_image_path(image_uri)
    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    return image_path


def _verify_input_checksum(image_path: Path, expected_sha256: str | None) -> None:
    if not expected_sha256:
        return
    actual_sha256 = sha256_file(image_path)
    if actual_sha256.lower() != expected_sha256.strip().lower():
        raise ValueError(
            f"Input image checksum mismatch for {image_path}: "
            f"expected {expected_sha256}, got {actual_sha256}."
        )


def run_inference_pipeline(
    request: InferenceRequest,
    *,
    store: ObjectStore | None = None,
    device: str = "cpu",
    roi_model_version: str = DEFAULT_ROI_MODEL_VERSION,
    feature_ae_version: str = DEFAULT_FEATURE_AE_MODEL_VERSION,
) -> InferencePipelineResult:
    object_store = store or create_object_store()

    with tempfile.TemporaryDirectory(prefix="iqa_inference_") as tmp:
        work_dir = Path(tmp)
        image_path = _resolve_input_image(
            request.image_uri,
            work_dir=work_dir,
            store=object_store,
        )
        _verify_input_checksum(image_path, request.sha256)

        roi_checkpoint = resolve_roi_segmenter_checkpoint(
            version=roi_model_version,
            strict_checksum=True,
        )
        feature_checkpoint = resolve_feature_ae_checkpoint(
            version=feature_ae_version,
            strict_checksum=True,
        )
        thresholds = load_feature_ae_decision_thresholds(feature_ae_version)
        reference_contract = load_feature_ae_reference_contract(feature_ae_version)

        threshold_orange = float(thresholds["threshold_orange"])
        threshold_red = float(thresholds["threshold_red"])

        roi_mask_path = work_dir / "roi_mask.png"
        roi_probability_path = work_dir / "roi_probability.png"
        heatmap_path = work_dir / "heatmap.png"

        roi = predict_roi_image(
            image_path,
            roi_checkpoint,
            threshold=reference_contract.roi_threshold,
            device=device,
            output_mask=roi_mask_path,
            output_probability_map=roi_probability_path,
        )

        feature = predict_feature_ae_image(
            image_path,
            feature_checkpoint,
            image_size=reference_contract.tile_size,
            context_size=reference_contract.context_size,
            threshold_orange=threshold_orange,
            threshold_red=threshold_red,
            threshold_source="manifest",
            roi_mask_path=roi_mask_path,
            roi_probability_path=roi_probability_path,
            score_smoothing=reference_contract.score_smoothing,
            score_image=reference_contract.score_image,
            topk_fraction=reference_contract.topk_fraction,
            heatmap_output_path=heatmap_path,
            device=device,
            pretrained_teacher=True,
            layers=reference_contract.layers,
            reference_contract=reference_contract,
        )

        context = VisualArtifactContext(
            scenario_id=request.scenario_id,
            lot_id=request.lot_id or "unknown_lot",
            piece_event_id=request.piece_event_id,
            image_id=_image_id(request.image_uri),
        )
        roi_mask_uri = publish_roi_mask(
            roi_mask_path,
            context,
            store=object_store,
        )
        heatmap_uri = publish_heatmap(
            heatmap_path,
            context,
            store=object_store,
        )

        roi_artifact = RoiPredictionArtifact(
            piece_event_id=request.piece_event_id,
            image_id=context.image_id,
            image_uri=request.image_uri,
            roi_mask_uri=roi_mask_uri,
            roi_model_version=roi_model_version,
            roi_ratio=roi.roi_ratio,
            roi_quality_status=roi.roi_quality_status,
            source=request.source_class or "runtime_inference",
            scenario_id=request.scenario_id,
            dataset_version=request.dataset_version or "unknown_dataset",
        )
        result = InferenceResult(
            piece_event_id=request.piece_event_id,
            scenario_id=request.scenario_id,
            score=float(feature.score),
            decision=decision_from_roi_and_score(
                roi.roi_quality_status,
                feature.score,
                orange_threshold=threshold_orange,
                red_threshold=threshold_red,
            ),
            heatmap_uri=heatmap_uri,
            roi_status=roi.roi_quality_status,
            roi_model_version=roi_model_version,
            feature_ae_version=feature_ae_version,
        )

        return InferencePipelineResult(
            request=request,
            roi_prediction=roi_artifact,
            result=result,
        )


__all__ = ["run_inference_pipeline"]
