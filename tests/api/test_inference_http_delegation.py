"""Tests for the HTTP inference client adapter and its contract validation.

These exercise ``HttpInferenceClient`` directly: transport, HTTP-error mapping
and inference-contract validation, without starting FastAPI or a TestClient.
"""

from __future__ import annotations

import json
import socket
from io import BytesIO
from urllib.error import HTTPError, URLError

import pytest

from iqa.inference import client as inference_client
from iqa.inference.client import HttpInferenceClient, InferenceClientError
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


def test_http_client_returns_valid_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IQA_INFERENCE_URL", "http://iqa-inference:8100")
    monkeypatch.setattr(inference_client, "urlopen", lambda request, timeout: FakeResponse(_valid_payload()))

    result = HttpInferenceClient().predict(_request())

    assert result.score == 150.0
    assert result.decision == "Orange"
    assert result.roi_status == "ok"
    assert result.heatmap_uri.startswith("s3://iqa-heatmaps/")


def test_http_client_maps_connection_failure_to_503(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(request, timeout):
        raise URLError("connection refused")

    monkeypatch.setattr(inference_client, "urlopen", fail)

    with pytest.raises(InferenceClientError) as caught:
        HttpInferenceClient().predict(_request())

    assert caught.value.status_code == 503
    assert caught.value.error_code == "inference_service_unavailable"


def test_http_client_maps_timeout_to_504(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(request, timeout):
        raise socket.timeout("timed out")

    monkeypatch.setattr(inference_client, "urlopen", fail)

    with pytest.raises(InferenceClientError) as caught:
        HttpInferenceClient().predict(_request())

    assert caught.value.status_code == 504
    assert caught.value.error_code == "inference_service_timeout"


def test_http_client_maps_invalid_json_to_502(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(inference_client, "urlopen", lambda request, timeout: FakeResponse(b"not-json"))

    with pytest.raises(InferenceClientError) as caught:
        HttpInferenceClient().predict(_request())

    assert caught.value.status_code == 502
    assert caught.value.error_code == "invalid_inference_response"


def test_http_client_maps_invalid_contract_to_502(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(inference_client, "urlopen", lambda request, timeout: FakeResponse({"score": 150.0}))

    with pytest.raises(InferenceClientError) as caught:
        HttpInferenceClient().predict(_request())

    assert caught.value.status_code == 502
    assert caught.value.error_code == "invalid_inference_response"


def test_http_client_rejects_invalid_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _valid_payload()
    payload["decision"] = "Bleu"
    monkeypatch.setattr(inference_client, "urlopen", lambda request, timeout: FakeResponse(payload))

    with pytest.raises(InferenceClientError) as caught:
        HttpInferenceClient().predict(_request())

    assert caught.value.status_code == 502
    assert caught.value.error_code == "invalid_inference_response"
    assert "decision" in (caught.value.reason or "").lower()


def test_http_client_rejects_traceability_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _valid_payload()
    payload["piece_event_id"] = "piece_other"
    monkeypatch.setattr(inference_client, "urlopen", lambda request, timeout: FakeResponse(payload))

    with pytest.raises(InferenceClientError) as caught:
        HttpInferenceClient().predict(_request())

    assert caught.value.status_code == 502
    assert caught.value.error_code == "invalid_inference_response"


def test_http_client_preserves_404_from_inference(
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

    monkeypatch.setattr(inference_client, "urlopen", fail)

    with pytest.raises(InferenceClientError) as caught:
        HttpInferenceClient().predict(_request())

    assert caught.value.status_code == 404
    assert caught.value.error_code == "inference_input_not_found"


def test_http_client_preserves_422_from_inference(
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

    monkeypatch.setattr(inference_client, "urlopen", fail)

    with pytest.raises(InferenceClientError) as caught:
        HttpInferenceClient().predict(_request())

    assert caught.value.status_code == 422
    assert caught.value.error_code == "inference_input_invalid"
