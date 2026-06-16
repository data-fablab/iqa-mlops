"""NAT13 API error standardization contracts."""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError

from iqa.api.main import (
    ADMIN_RELOAD_LOG,
    AI_SECURITY_METRICS,
    DISPLAY_FEEDBACK_STORE,
    FEEDBACK_STORE,
    PREDICTION_STORE,
    _api_error_detail,
    _request_validation_error_handler,
    feedback,
    predict,
    reload_model,
)
from iqa.api.schemas import FeedbackRequest, PredictRequest, ReloadModelRequest


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch: pytest.MonkeyPatch) -> None:
    PREDICTION_STORE.clear()
    FEEDBACK_STORE.clear()
    DISPLAY_FEEDBACK_STORE.clear()
    ADMIN_RELOAD_LOG.clear()
    for key in AI_SECURITY_METRICS:
        AI_SECURITY_METRICS[key] = 0
    monkeypatch.delenv("IQA_SERVICE_TOKEN", raising=False)
    monkeypatch.delenv("IQA_ADMIN_TOKEN", raising=False)
    yield
    PREDICTION_STORE.clear()
    FEEDBACK_STORE.clear()
    DISPLAY_FEEDBACK_STORE.clear()
    ADMIN_RELOAD_LOG.clear()


def _assert_error_detail(
    detail: dict[str, object],
    *,
    status_code: int,
    error_code: str,
    message: str,
) -> None:
    assert detail["status_code"] == status_code
    assert detail["error_code"] == error_code
    assert detail["message"] == message


def test_nat13_standardizes_404_prediction_not_found_error() -> None:
    with pytest.raises(HTTPException) as exc_info:
        feedback(
            FeedbackRequest(
                prediction_id="pred_missing_nat13",
                piece_event_id="piece_nat13",
                scenario_id="scenario_nat13",
            )
        )

    assert exc_info.value.status_code == 404
    _assert_error_detail(
        exc_info.value.detail,
        status_code=404,
        error_code="prediction_not_found",
        message="Unknown prediction_id.",
    )
    assert exc_info.value.detail["incident_type"] == "invalid_prediction_request"


def test_nat13_standardizes_409_feedback_conflict_error() -> None:
    response = predict(
        PredictRequest(
            piece_event_id="piece_nat13",
            scenario_id="scenario_nat13",
            image_uri="s3://iqa/raw/piece_nat13.png",
        )
    )
    prediction_id = response["prediction"]["prediction_id"]

    with pytest.raises(HTTPException) as exc_info:
        feedback(
            FeedbackRequest(
                prediction_id=prediction_id,
                piece_event_id="piece_nat13_attacker",
                scenario_id="scenario_nat13",
            )
        )

    assert exc_info.value.status_code == 409
    _assert_error_detail(
        exc_info.value.detail,
        status_code=409,
        error_code="feedback_piece_event_mismatch",
        message="prediction_id does not match piece_event_id.",
    )
    assert exc_info.value.detail["incident_type"] == "feedback_conflict"


def test_nat13_standardizes_400_unknown_feedback_source_error() -> None:
    response = predict(
        PredictRequest(
            piece_event_id="piece_nat13_source",
            scenario_id="scenario_nat13",
            image_uri="s3://iqa/raw/piece_nat13_source.png",
        )
    )
    prediction_id = response["prediction"]["prediction_id"]

    request = FeedbackRequest.model_construct(
        prediction_id=prediction_id,
        piece_event_id="piece_nat13_source",
        scenario_id="scenario_nat13",
        feedback_source="attacker_source",
        gt_mask_has_defect=False,
    )

    with pytest.raises(HTTPException) as exc_info:
        feedback(request)

    assert exc_info.value.status_code == 400
    _assert_error_detail(
        exc_info.value.detail,
        status_code=400,
        error_code="unknown_feedback_source",
        message="Unknown feedback_source.",
    )


def test_nat13_standardizes_401_service_token_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IQA_SERVICE_TOKEN", "secret")
    response = predict(
        PredictRequest(
            piece_event_id="piece_nat13_token",
            scenario_id="scenario_nat13",
            image_uri="s3://iqa/raw/piece_nat13_token.png",
        )
    )
    prediction_id = response["prediction"]["prediction_id"]

    with pytest.raises(HTTPException) as exc_info:
        feedback(
            FeedbackRequest(
                prediction_id=prediction_id,
                piece_event_id="piece_nat13_token",
                scenario_id="scenario_nat13",
            ),
            x_iqa_service_token="bad",
        )

    assert exc_info.value.status_code == 401
    _assert_error_detail(
        exc_info.value.detail,
        status_code=401,
        error_code="invalid_token",
        message="Missing or invalid IQA_SERVICE_TOKEN.",
    )


def test_nat13_standardizes_admin_reload_refusal_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(HTTPException) as missing_token:
        reload_model(ReloadModelRequest(scenario_id="scenario_nat13"), x_iqa_admin_token="secret")

    assert missing_token.value.status_code == 503
    _assert_error_detail(
        missing_token.value.detail,
        status_code=503,
        error_code="admin_token_not_configured",
        message="IQA_ADMIN_TOKEN is not configured.",
    )
    assert missing_token.value.detail["audit_logged"] is True
    assert "reload_event_id" in missing_token.value.detail

    monkeypatch.setenv("IQA_ADMIN_TOKEN", "secret")

    with pytest.raises(HTTPException) as invalid_token:
        reload_model(ReloadModelRequest(scenario_id="scenario_nat13"), x_iqa_admin_token="bad")

    assert invalid_token.value.status_code == 401
    _assert_error_detail(
        invalid_token.value.detail,
        status_code=401,
        error_code="invalid_admin_token",
        message="Missing or invalid IQA_ADMIN_TOKEN.",
    )
    assert invalid_token.value.detail["audit_logged"] is True
    assert "reload_event_id" in invalid_token.value.detail


def test_nat13_standardizes_422_request_validation_error() -> None:
    class DummyUrl:
        path = "/predict"

        def __str__(self) -> str:
            return "http://testserver/predict"

    class DummyRequest:
        url = DummyUrl()

    exc = RequestValidationError(
        [
            {
                "type": "missing",
                "loc": ("body", "scenario_id"),
                "msg": "Field required",
                "input": {
                    "piece_event_id": "piece_nat13_422",
                    "image_uri": "s3://iqa/raw/piece_nat13_422.png",
                },
            }
        ]
    )

    response = asyncio.run(_request_validation_error_handler(DummyRequest(), exc))
    payload = json.loads(response.body.decode("utf-8"))
    detail = payload["detail"]

    assert response.status_code == 422
    _assert_error_detail(
        detail,
        status_code=422,
        error_code="validation_error",
        message="Request validation failed.",
    )
    assert detail["incident_type"] == "invalid_prediction_request"
    assert detail["details"]["path"] == "/predict"


def test_nat13_error_schema_covers_403_and_500_contracts() -> None:
    forbidden = _api_error_detail(
        status_code=403,
        error_code="forbidden",
        message="Forbidden.",
        reason="Action is not allowed.",
    )
    internal = _api_error_detail(
        status_code=500,
        error_code="internal_server_error",
        message="Internal server error.",
        reason="Unhandled server exception.",
    )

    _assert_error_detail(forbidden, status_code=403, error_code="forbidden", message="Forbidden.")
    _assert_error_detail(
        internal,
        status_code=500,
        error_code="internal_server_error",
        message="Internal server error.",
    )
