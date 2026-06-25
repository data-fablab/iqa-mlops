"""Tests for the real inference service boundary."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from iqa.inference import service
from iqa.inference.contracts import InferenceRequest, InferenceResult


def _result(request: InferenceRequest) -> InferenceResult:
    return InferenceResult(
        piece_event_id=request.piece_event_id,
        scenario_id=request.scenario_id,
        score=150.0,
        decision="Orange",
        heatmap_uri="s3://iqa-heatmaps/lots/demo/heatmap.png",
        roi_status="ok",
        roi_model_version="roi_segmenter_v001_fixed",
        feature_ae_version="rd_feature_ae_gated_v001_bootstrap",
    )


def test_predict_delegates_to_real_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, InferenceRequest] = {}

    def fake_real_inference(request: InferenceRequest) -> InferenceResult:
        captured["request"] = request
        return _result(request)

    monkeypatch.setattr(
        service,
        "_run_real_inference",
        fake_real_inference,
        raising=False,
    )

    response = service.predict(
        service.InferenceServiceRequest(
            piece_event_id="piece_001",
            scenario_id="production_replay_natural",
            image_uri="s3://iqa-ingested-images/incoming/piece.jpg",
            sha256="a" * 64,
            lot_id="LOT_001",
            source_class="casting",
            dataset_version="casting_v001",
        )
    )

    request = captured["request"]

    assert request.piece_event_id == "piece_001"
    assert request.sha256 == "a" * 64
    assert request.lot_id == "LOT_001"
    assert request.source_class == "casting"
    assert request.dataset_version == "casting_v001"
    assert response["score"] == 150.0
    assert response["decision"] == "Orange"
    assert response["roi_status"] == "ok"
    assert response["heatmap_uri"].startswith("s3://iqa-heatmaps/")


def test_predict_maps_missing_input_image_to_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_real_inference(request: InferenceRequest) -> InferenceResult:
        raise FileNotFoundError(f"Input image not found: {request.image_uri}")

    monkeypatch.setattr(
        service,
        "_run_real_inference",
        fake_real_inference,
        raising=False,
    )

    with pytest.raises(HTTPException) as caught:
        service.predict(
            service.InferenceServiceRequest(
                piece_event_id="piece_missing",
                image_uri="/missing/piece.jpg",
            )
        )

    assert caught.value.status_code == 404
    assert "Input image not found" in str(caught.value.detail)


def test_predict_maps_input_checksum_mismatch_to_422(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_real_inference(request: InferenceRequest) -> InferenceResult:
        raise ValueError(
            f"Input image checksum mismatch for {request.image_uri}"
        )

    monkeypatch.setattr(
        service,
        "_run_real_inference",
        fake_real_inference,
        raising=False,
    )

    with pytest.raises(HTTPException) as caught:
        service.predict(
            service.InferenceServiceRequest(
                piece_event_id="piece_bad_sha",
                image_uri="/data/piece.jpg",
                sha256="0" * 64,
            )
        )

    assert caught.value.status_code == 422
    assert "checksum mismatch" in str(caught.value.detail)


def test_predict_maps_model_failure_to_503(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_real_inference(request: InferenceRequest) -> InferenceResult:
        raise FileNotFoundError("Model artifact not found: checkpoint.pt")

    monkeypatch.setattr(
        service,
        "_run_real_inference",
        fake_real_inference,
        raising=False,
    )

    with pytest.raises(HTTPException) as caught:
        service.predict(
            service.InferenceServiceRequest(
                piece_event_id="piece_model_error",
                image_uri="/data/piece.jpg",
            )
        )

    assert caught.value.status_code == 503
    assert "Inference unavailable" in str(caught.value.detail)


def test_service_predict_path_does_not_use_placeholder() -> None:
    source = Path(service.__file__).read_text(encoding="utf-8")

    assert "placeholder_inference" not in source
