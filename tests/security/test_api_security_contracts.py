from __future__ import annotations

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from iqa.api.main import (
    ADMIN_RELOAD_LOG,
    AI_SECURITY_METRICS,
    FeedbackRequest,
    PieceEventPredictRequest,
    PredictRequest,
    ReloadModelRequest,
    feedback,
    metrics,
    predict,
    predict_piece_event,
    reload_model,
)


@pytest.fixture(autouse=True)
def _clear_service_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IQA_SERVICE_TOKEN", raising=False)


def _reset_security_metrics() -> None:
    for key in AI_SECURITY_METRICS:
        AI_SECURITY_METRICS[key] = 0


def test_prediction_contract_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        PredictRequest(
            piece_event_id="piece-contract-extra",
            scenario_id="demo",
            image_uri="s3://bucket/key.jpg",
            unexpected_field="forbidden",
        )


def test_piece_event_predict_contract_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        PieceEventPredictRequest(
            scenario_id="demo",
            image_uri="s3://bucket/key.jpg",
            unexpected_field="forbidden",
        )


def test_feedback_contract_requires_prediction_id() -> None:
    with pytest.raises(ValidationError):
        FeedbackRequest(
            piece_event_id="piece-no-prediction-id",
            scenario_id="demo",
            gt_mask_has_defect=True,
        )


def test_prediction_security_audit_fields_are_returned() -> None:
    response = predict_piece_event(
        "piece-security-audit",
        PieceEventPredictRequest(scenario_id="demo", image_uri="s3://bucket/key.jpg"),
    )

    assert response["prediction"]["prediction_id"].startswith("pred_")
    assert response["prediction"]["audit_logged"] is True
    assert response["audit"]["audit_logged"] is True
    assert response["audit"]["piece_event_id"] == "piece-security-audit"
    assert response["audit"]["scenario_id"] == "demo"
    assert response["audit"]["audit_sink"] == "api_response_mvp"


def test_feedback_security_rejects_replayed_prediction() -> None:
    prediction_response = predict(
        PredictRequest(
            piece_event_id="piece-security-replay",
            scenario_id="demo",
            image_uri="s3://bucket/key.jpg",
        )
    )
    prediction_id = prediction_response["prediction"]["prediction_id"]

    first_response = feedback(
        FeedbackRequest(
            prediction_id=prediction_id,
            piece_event_id="piece-security-replay",
            scenario_id="demo",
            gt_mask_has_defect=True,
        )
    )

    assert first_response["accepted"] is True
    assert first_response["feedback_closed"] is True

    with pytest.raises(HTTPException) as exc_info:
        feedback(
            FeedbackRequest(
                prediction_id=prediction_id,
                piece_event_id="piece-security-replay",
                scenario_id="demo",
                gt_mask_has_defect=True,
            )
        )

    assert exc_info.value.status_code == 409


def test_feedback_security_conflict_increments_metrics() -> None:
    _reset_security_metrics()

    prediction_response = predict(
        PredictRequest(
            piece_event_id="piece-security-conflict",
            scenario_id="demo",
            image_uri="s3://bucket/key.jpg",
        )
    )
    prediction_id = prediction_response["prediction"]["prediction_id"]

    with pytest.raises(HTTPException) as exc_info:
        feedback(
            FeedbackRequest(
                prediction_id=prediction_id,
                piece_event_id="piece-security-conflict-other",
                scenario_id="demo",
                gt_mask_has_defect=True,
            )
        )

    assert exc_info.value.status_code == 409

    body = metrics()
    assert "iqa_feedback_conflict_total 1" in body
    assert "iqa_ai_security_incident_total 1" in body


def test_human_feedback_is_display_only_and_not_train_eligible() -> None:
    prediction_response = predict(
        PredictRequest(
            piece_event_id="piece-security-human",
            scenario_id="demo",
            image_uri="s3://bucket/key.jpg",
        )
    )
    prediction_id = prediction_response["prediction"]["prediction_id"]

    response = feedback(
        FeedbackRequest(
            prediction_id=prediction_id,
            piece_event_id="piece-security-human",
            scenario_id="demo",
            feedback_source="human_sophie",
        )
    )

    assert response["accepted"] is True
    assert response["feedback_closed"] is False
    assert response["display_decision_source"] == "human_sophie"
    assert response["train_eligibility_source"] == "oracle_gt"
    assert response["eligible_for_train"] is False


def test_admin_reload_security_logs_refused_without_configured_token(monkeypatch: pytest.MonkeyPatch) -> None:
    ADMIN_RELOAD_LOG.clear()
    _reset_security_metrics()
    monkeypatch.delenv("IQA_ADMIN_TOKEN", raising=False)

    with pytest.raises(HTTPException) as exc_info:
        reload_model(ReloadModelRequest(scenario_id="demo"), x_iqa_admin_token="secret")

    assert exc_info.value.status_code == 503
    assert ADMIN_RELOAD_LOG[-1]["reload_status"] == "refused"
    assert ADMIN_RELOAD_LOG[-1]["accepted"] is False

    body = metrics()
    assert "iqa_reload_refused_total 1" in body
    assert "iqa_ai_security_incident_total 1" in body


def test_admin_reload_security_logs_refused_with_invalid_token(monkeypatch: pytest.MonkeyPatch) -> None:
    ADMIN_RELOAD_LOG.clear()
    _reset_security_metrics()
    monkeypatch.setenv("IQA_ADMIN_TOKEN", "secret")

    with pytest.raises(HTTPException) as exc_info:
        reload_model(ReloadModelRequest(scenario_id="demo"), x_iqa_admin_token="bad")

    assert exc_info.value.status_code == 401
    assert ADMIN_RELOAD_LOG[-1]["reload_status"] == "refused"
    assert ADMIN_RELOAD_LOG[-1]["accepted"] is False

    body = metrics()
    assert "iqa_reload_refused_total 1" in body
    assert "iqa_ai_security_incident_total 1" in body


def test_admin_reload_security_accepts_valid_token_and_writes_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    ADMIN_RELOAD_LOG.clear()
    monkeypatch.setenv("IQA_ADMIN_TOKEN", "secret")

    response = reload_model(ReloadModelRequest(scenario_id="demo"), x_iqa_admin_token="secret")

    assert response["accepted"] is True
    assert response["reload_status"] == "accepted"
    assert response["audit_logged"] is True
    assert response["audit"]["reload_status"] == "accepted"
    assert response["audit"]["scenario_id"] == "demo"
    assert ADMIN_RELOAD_LOG[-1]["reload_status"] == "accepted"
