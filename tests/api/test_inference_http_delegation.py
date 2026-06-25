"""Tests for API delegation to the inference service."""

from __future__ import annotations

import json
import socket
from io import BytesIO
from urllib.error import HTTPError, URLError

import pytest
from fastapi import HTTPException

from iqa.api import main as api
from iqa.inference.contracts import InferenceRequest


class FakeResponse:
    def __init__(self, payload: dict | bytes) -> None:
        self.payload = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def _request() -> InferenceRequest:
    return InferenceRequest(
        piece_event_id="piece_http_001",
        scenario_id="production_replay_natural",
        image_uri="s3://iqa-ingested-images/incoming/piece.jpg",
        sha256="a" * 64,
        lot_id="LOT_001",
        source_class="casting",
        dataset_version="casting_v001",
    )


def _valid_payload() -> dict:
    return {
        "piece_event_id": "piece_http_001",
        "scenario_id": "production_replay_natural",
        "score": 150.0,
        "decision": "Orange",
        "heatmap_uri": "s3://iqa-heatmaps/lots/demo/heatmap.png",
        "roi_status": "ok",
        "roi_model_version": "roi_segmenter_v001_fixed",
        "feature_ae_version": "rd_feature_ae_gated_v001_bootstrap",
    }


def test_call_inference_service_returns_valid_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IQA_INFERENCE_URL", "http://iqa-inference:8100")
    monkeypatch.setattr(api, "urlopen", lambda request, timeout: FakeResponse(_valid_payload()))

    result = api._call_inference_service(_request())

    assert result.score == 150.0
    assert result.decision == "Orange"
    assert result.roi_status == "ok"
    assert result.heatmap_uri.startswith("s3://iqa-heatmaps/")


def test_call_inference_service_maps_connection_failure_to_503(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(request, timeout):
        raise URLError("connection refused")

    monkeypatch.setattr(api, "urlopen", fail)

    with pytest.raises(HTTPException) as caught:
        api._call_inference_service(_request())

    assert caught.value.status_code == 503
    assert caught.value.detail["error_code"] == "inference_service_unavailable"


def test_call_inference_service_maps_timeout_to_504(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(request, timeout):
        raise socket.timeout("timed out")

    monkeypatch.setattr(api, "urlopen", fail)

    with pytest.raises(HTTPException) as caught:
        api._call_inference_service(_request())

    assert caught.value.status_code == 504
    assert caught.value.detail["error_code"] == "inference_service_timeout"


def test_call_inference_service_maps_invalid_json_to_502(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api, "urlopen", lambda request, timeout: FakeResponse(b"not-json"))

    with pytest.raises(HTTPException) as caught:
        api._call_inference_service(_request())

    assert caught.value.status_code == 502
    assert caught.value.detail["error_code"] == "invalid_inference_response"


def test_call_inference_service_maps_invalid_contract_to_502(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api, "urlopen", lambda request, timeout: FakeResponse({"score": 150.0}))

    with pytest.raises(HTTPException) as caught:
        api._call_inference_service(_request())

    assert caught.value.status_code == 502
    assert caught.value.detail["error_code"] == "invalid_inference_response"


def test_call_inference_service_preserves_404_from_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.dumps({"detail": "Input image not found"}).encode("utf-8")

    def fail(request, timeout):
        raise HTTPError(
            request.full_url,
            404,
            "Not Found",
            {},
            BytesIO(payload),
        )

    monkeypatch.setattr(api, "urlopen", fail)

    with pytest.raises(HTTPException) as caught:
        api._call_inference_service(_request())

    assert caught.value.status_code == 404
    assert caught.value.detail["error_code"] == "inference_input_not_found"


def test_call_inference_service_preserves_422_from_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.dumps({"detail": "Input image checksum mismatch"}).encode("utf-8")

    def fail(request, timeout):
        raise HTTPError(
            request.full_url,
            422,
            "Unprocessable Entity",
            {},
            BytesIO(payload),
        )

    monkeypatch.setattr(api, "urlopen", fail)

    with pytest.raises(HTTPException) as caught:
        api._call_inference_service(_request())

    assert caught.value.status_code == 422
    assert caught.value.detail["error_code"] == "inference_input_invalid"
