"""Inference client seam: the API's port to the iqa-inference service.

Two adapters live behind the ``InferenceClient`` interface:

- ``HttpInferenceClient`` performs HTTP transport, HTTP-error mapping and
  inference-contract validation against the separate iqa-inference service.
- ``StubInferenceClient`` returns a deterministic result for tests and demos.

Contract validation (required fields, ``decision`` in {Vert, Orange, Rouge},
traceability echo of ``piece_event_id``/``scenario_id``) is the invariant of the
interface: it can be exercised without HTTP or FastAPI.

Per ADR 0007/0008 the API and inference runtimes stay separate; this module is
only the client the API uses to reach the inference service, never the runtime.
"""

from __future__ import annotations

import json
import os
import socket
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request as UrlRequest, urlopen

from iqa.inference.contracts import InferenceRequest, InferenceResult


DEFAULT_INFERENCE_URL = "http://iqa-inference:8100"
DEFAULT_INFERENCE_TIMEOUT_SECONDS = "120"
INFERENCE_CLIENT_ENV = "IQA_INFERENCE_CLIENT"
VALID_DECISIONS: frozenset[str] = frozenset({"Vert", "Orange", "Rouge"})
REQUIRED_RESPONSE_FIELDS: frozenset[str] = frozenset(
    {
        "piece_event_id",
        "scenario_id",
        "score",
        "decision",
        "heatmap_uri",
        "roi_status",
        "roi_model_version",
        "feature_ae_version",
    }
)


class InferenceClientError(Exception):
    """Raised when the inference call fails or its contract is violated.

    Carries the standardized API error fields so the gateway can translate it
    into an HTTP response without re-deriving status codes or error codes.
    """

    def __init__(
        self,
        *,
        status_code: int,
        error_code: str,
        message: str,
        reason: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.message = message
        self.reason = reason
        self.details = details or {}


class InferenceClient(Protocol):
    """Interface the API depends on to obtain an inference result."""

    def predict(self, request: InferenceRequest) -> InferenceResult: ...


def _error(
    status_code: int,
    error_code: str,
    message: str,
    reason: str | None = None,
    details: dict[str, Any] | None = None,
) -> InferenceClientError:
    return InferenceClientError(
        status_code=status_code,
        error_code=error_code,
        message=message,
        reason=reason,
        details=details,
    )


class HttpInferenceClient:
    """HTTP adapter to the separate iqa-inference service.

    Configuration (base URL, timeout) is read from the environment at call time
    unless explicit overrides are supplied, so the deployment can retarget the
    inference service without rebuilding the client.
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout_seconds: float | str | None = None,
    ) -> None:
        self._base_url_override = base_url
        self._timeout_override = timeout_seconds

    def _resolve_base_url(self) -> str:
        raw = (
            self._base_url_override
            if self._base_url_override is not None
            else os.environ.get("IQA_INFERENCE_URL", DEFAULT_INFERENCE_URL)
        )
        base_url = raw.strip().rstrip("/")
        if not base_url:
            raise _error(
                503,
                "inference_service_configuration_error",
                "Inference service URL is not configured.",
                "IQA_INFERENCE_URL is empty.",
                {"path": "/predict"},
            )
        return base_url

    def _resolve_timeout(self) -> float:
        raw = (
            self._timeout_override
            if self._timeout_override is not None
            else os.environ.get("IQA_INFERENCE_TIMEOUT_SECONDS", DEFAULT_INFERENCE_TIMEOUT_SECONDS)
        )
        try:
            timeout = float(raw)
        except (TypeError, ValueError):
            raise _error(
                503,
                "inference_service_configuration_error",
                "Inference service timeout is invalid.",
                f"Invalid IQA_INFERENCE_TIMEOUT_SECONDS: {raw!r}.",
                {"path": "/predict"},
            ) from None
        if timeout <= 0:
            raise _error(
                503,
                "inference_service_configuration_error",
                "Inference service timeout is invalid.",
                "IQA_INFERENCE_TIMEOUT_SECONDS must be greater than zero.",
                {"path": "/predict"},
            )
        return timeout

    @staticmethod
    def _http_error_detail(error: HTTPError) -> str:
        try:
            payload = json.loads(error.read().decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return str(error.reason or f"HTTP {error.code}")

        if isinstance(payload, dict):
            detail = payload.get("detail")
            if isinstance(detail, str):
                return detail
            if detail is not None:
                return json.dumps(detail, ensure_ascii=False)

        return str(error.reason or f"HTTP {error.code}")

    def predict(self, request: InferenceRequest) -> InferenceResult:
        base_url = self._resolve_base_url()
        timeout = self._resolve_timeout()

        payload = {
            "piece_event_id": request.piece_event_id,
            "scenario_id": request.scenario_id,
            "image_uri": request.image_uri,
            "sha256": request.sha256,
            "lot_id": request.lot_id,
            "source_class": request.source_class,
            "dataset_version": request.dataset_version,
        }
        http_request = UrlRequest(
            f"{base_url}/predict",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(http_request, timeout=timeout) as response:
                raw_response = response.read()
        except HTTPError as error:
            raise self._mapped_http_error(error) from error
        except socket.timeout as error:
            raise _error(
                504,
                "inference_service_timeout",
                "Inference service timed out.",
                str(error),
                {"path": "/predict"},
            ) from error
        except URLError as error:
            if isinstance(error.reason, (socket.timeout, TimeoutError)):
                raise _error(
                    504,
                    "inference_service_timeout",
                    "Inference service timed out.",
                    str(error.reason),
                    {"path": "/predict"},
                ) from error
            raise _error(
                503,
                "inference_service_unavailable",
                "Inference service is unavailable.",
                str(error.reason),
                {"path": "/predict"},
            ) from error

        return self._parse_response(raw_response, request)

    def _mapped_http_error(self, error: HTTPError) -> InferenceClientError:
        detail = self._http_error_detail(error)
        if error.code == 404:
            return _error(
                404,
                "inference_input_not_found",
                "Inference input image was not found.",
                detail,
                {"path": "/predict", "upstream_status": error.code},
            )
        if error.code == 422:
            return _error(
                422,
                "inference_input_invalid",
                "Inference input was rejected.",
                detail,
                {"path": "/predict", "upstream_status": error.code},
            )
        if error.code == 503:
            return _error(
                503,
                "inference_service_unavailable",
                "Inference service is unavailable.",
                detail,
                {"path": "/predict", "upstream_status": error.code},
            )
        return _error(
            502,
            "invalid_inference_response",
            "Inference service returned an unexpected HTTP response.",
            detail,
            {"path": "/predict", "upstream_status": error.code},
        )

    @staticmethod
    def _parse_response(raw_response: bytes, request: InferenceRequest) -> InferenceResult:
        try:
            response_payload = json.loads(raw_response.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise _error(
                502,
                "invalid_inference_response",
                "Inference service returned invalid JSON.",
                str(error),
                {"path": "/predict"},
            ) from error

        if not isinstance(response_payload, dict):
            raise _error(
                502,
                "invalid_inference_response",
                "Inference service returned an invalid contract.",
                "Expected a JSON object.",
                {"path": "/predict"},
            )

        missing = sorted(REQUIRED_RESPONSE_FIELDS - set(response_payload))
        if missing:
            raise _error(
                502,
                "invalid_inference_response",
                "Inference service returned an incomplete contract.",
                f"Missing fields: {', '.join(missing)}.",
                {"path": "/predict"},
            )

        decision = response_payload["decision"]
        if decision not in VALID_DECISIONS:
            raise _error(
                502,
                "invalid_inference_response",
                "Inference service returned an invalid decision.",
                f"Unsupported decision: {decision!r}.",
                {"path": "/predict"},
            )

        if (
            response_payload["piece_event_id"] != request.piece_event_id
            or response_payload["scenario_id"] != request.scenario_id
        ):
            raise _error(
                502,
                "invalid_inference_response",
                "Inference service returned mismatched traceability identifiers.",
                "piece_event_id or scenario_id does not match the request.",
                {"path": "/predict"},
            )

        try:
            return InferenceResult(
                piece_event_id=str(response_payload["piece_event_id"]),
                scenario_id=str(response_payload["scenario_id"]),
                score=float(response_payload["score"]),
                decision=decision,
                heatmap_uri=(
                    None
                    if response_payload["heatmap_uri"] is None
                    else str(response_payload["heatmap_uri"])
                ),
                roi_status=(
                    None
                    if response_payload["roi_status"] is None
                    else str(response_payload["roi_status"])
                ),
                roi_model_version=str(response_payload["roi_model_version"]),
                feature_ae_version=str(response_payload["feature_ae_version"]),
            )
        except (TypeError, ValueError) as error:
            raise _error(
                502,
                "invalid_inference_response",
                "Inference service returned an invalid contract.",
                str(error),
                {"path": "/predict"},
            ) from error


class StubInferenceClient:
    """Deterministic adapter for tests and demos (no HTTP, no FastAPI).

    Echoes the request traceability identifiers and returns the configured
    decision so callers exercise the same contract as the HTTP adapter.
    """

    def __init__(
        self,
        *,
        score: float = 0.0,
        decision: str = "Vert",
        heatmap_uri: str | None = None,
        roi_status: str | None = "ok",
        roi_model_version: str = "roi_segmenter_v001_fixed",
        feature_ae_version: str = "rd_feature_ae_gated_v001_bootstrap",
    ) -> None:
        self._score = score
        self._decision = decision
        self._heatmap_uri = heatmap_uri
        self._roi_status = roi_status
        self._roi_model_version = roi_model_version
        self._feature_ae_version = feature_ae_version

    def predict(self, request: InferenceRequest) -> InferenceResult:
        return InferenceResult(
            piece_event_id=request.piece_event_id,
            scenario_id=request.scenario_id,
            score=self._score,
            decision=self._decision,
            heatmap_uri=self._heatmap_uri,
            roi_status=self._roi_status,
            roi_model_version=self._roi_model_version,
            feature_ae_version=self._feature_ae_version,
        )


def create_inference_client() -> InferenceClient:
    """Resolve the configured inference client (env → HTTP by default)."""

    mode = os.environ.get(INFERENCE_CLIENT_ENV, "http").strip().lower()
    if mode == "stub":
        return StubInferenceClient()
    return HttpInferenceClient()


__all__ = [
    "HttpInferenceClient",
    "InferenceClient",
    "InferenceClientError",
    "StubInferenceClient",
    "create_inference_client",
]
