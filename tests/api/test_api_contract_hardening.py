"""NAT15 hardened API and error contract tests."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from iqa.api.main import (
    ADMIN_RELOAD_LOG,
    AI_SECURITY_METRICS,
    DISPLAY_FEEDBACK_STORE,
    FEEDBACK_STORE,
    INCIDENT_STORE,
    PREDICTION_STORE,
    _api_error_detail,
    feedback,
    list_incidents,
    metrics,
    predict,
    record_dataset_blocked_incident,
    reload_model,
)
from iqa.api.schemas import (
    ApiErrorResponse,
    FeedbackRequest,
    Incident,
    PredictRequest,
    ReloadModelRequest,
)


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch: pytest.MonkeyPatch) -> None:
    PREDICTION_STORE.clear()
    FEEDBACK_STORE.clear()
    DISPLAY_FEEDBACK_STORE.clear()
    INCIDENT_STORE.clear()
    ADMIN_RELOAD_LOG.clear()
    for key in AI_SECURITY_METRICS:
        AI_SECURITY_METRICS[key] = 0
    monkeypatch.delenv("IQA_SERVICE_TOKEN", raising=False)
    monkeypatch.delenv("IQA_ADMIN_TOKEN", raising=False)
    yield
    PREDICTION_STORE.clear()
    FEEDBACK_STORE.clear()
    DISPLAY_FEEDBACK_STORE.clear()
    INCIDENT_STORE.clear()
    ADMIN_RELOAD_LOG.clear()


def _create_prediction(
    *,
    piece_event_id: str = "piece_nat15",
    scenario_id: str = "scenario_nat15",
) -> str:
    response = predict(
        PredictRequest(
            piece_event_id=piece_event_id,
            scenario_id=scenario_id,
            image_uri=f"s3://iqa/raw/{piece_event_id}.png",
            sha256="6" * 64,
            lot_id="lot_nat15",
            source_class="Casting_class1",
            dataset_version="casting_v015",
        )
    )
    return response["prediction"]["prediction_id"]


def test_nat15_scenario_mismatch_error_creates_structured_conflict_incident() -> None:
    prediction_id = _create_prediction(
        piece_event_id="piece_nat15_scenario",
        scenario_id="scenario_nat15_expected",
    )

    with pytest.raises(HTTPException) as exc_info:
        feedback(
            FeedbackRequest(
                prediction_id=prediction_id,
                piece_event_id="piece_nat15_scenario",
                scenario_id="scenario_nat15_attacker",
            )
        )

    detail = exc_info.value.detail

    assert exc_info.value.status_code == 409
    assert detail["status_code"] == 409
    assert detail["error_code"] == "feedback_scenario_mismatch"
    assert detail["message"] == "prediction_id does not match scenario_id."
    assert detail["incident_type"] == "feedback_conflict"

    incidents = list_incidents(incident_type="feedback_conflict")
    assert len(incidents) == 1
    assert incidents[0]["scenario_id"] == "scenario_nat15_attacker"
    assert incidents[0]["metadata"]["expected_scenario_id"] == "scenario_nat15_expected"
    assert incidents[0]["metadata"]["received_scenario_id"] == "scenario_nat15_attacker"

    body = metrics()
    assert "iqa_feedback_conflict_total 1" in body
    assert "iqa_ai_security_incident_total 1" in body


def test_nat15_closed_feedback_error_keeps_standard_error_contract() -> None:
    prediction_id = _create_prediction(piece_event_id="piece_nat15_closed")

    first_response = feedback(
        FeedbackRequest(
            prediction_id=prediction_id,
            piece_event_id="piece_nat15_closed",
            scenario_id="scenario_nat15",
            feedback_source="oracle_gt",
            gt_mask_has_defect=False,
        )
    )

    assert first_response["accepted"] is True
    assert first_response["feedback_closed"] is True

    with pytest.raises(HTTPException) as exc_info:
        feedback(
            FeedbackRequest(
                prediction_id=prediction_id,
                piece_event_id="piece_nat15_closed",
                scenario_id="scenario_nat15",
                feedback_source="oracle_gt",
                gt_mask_has_defect=False,
            )
        )

    detail = exc_info.value.detail

    assert exc_info.value.status_code == 409
    assert detail["status_code"] == 409
    assert detail["error_code"] == "feedback_already_closed"
    assert detail["message"] == "Prediction already has a closed feedback."
    assert detail["incident_type"] == "invalid_prediction_request"


def test_nat15_service_token_accepts_matching_token_and_preserves_feedback_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IQA_SERVICE_TOKEN", "secret")
    prediction_id = _create_prediction(piece_event_id="piece_nat15_token")

    response = feedback(
        FeedbackRequest(
            prediction_id=prediction_id,
            piece_event_id="piece_nat15_token",
            scenario_id="scenario_nat15",
            feedback_source="oracle_gt",
            gt_mask_has_defect=False,
        ),
        x_iqa_service_token="secret",
    )

    assert response["accepted"] is True
    assert response["feedback_closed"] is True
    assert response["train_eligibility_source"] == "oracle_gt"
    assert response["eligible_for_train"] is True
    assert FEEDBACK_STORE[prediction_id]["feedback_source"] == "oracle_gt"


def test_nat15_admin_reload_success_does_not_create_reload_refused_incident(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IQA_ADMIN_TOKEN", "secret")

    response = reload_model(
        ReloadModelRequest(scenario_id="scenario_nat15", stage="prod"),
        x_iqa_admin_token="secret",
    )

    assert response["accepted"] is True
    assert response["reload_status"] == "accepted"
    assert response["audit_logged"] is True
    assert ADMIN_RELOAD_LOG[-1]["reload_status"] == "accepted"
    assert list_incidents(incident_type="reload_refused") == []


def test_nat15_error_detail_validates_against_api_error_response_schema() -> None:
    detail = _api_error_detail(
        status_code=500,
        error_code="internal_server_error",
        message="Internal server error.",
        reason="Unhandled server exception.",
        details={"path": "/predict"},
    )

    validated = ApiErrorResponse.model_validate(detail)

    assert validated.status_code == 500
    assert validated.error_code == "internal_server_error"
    assert validated.message == "Internal server error."
    assert validated.details["path"] == "/predict"
    assert "reload_event_id" not in detail


def test_nat15_incident_route_payloads_validate_against_incident_schema() -> None:
    incident = record_dataset_blocked_incident(
        scenario_id="scenario_nat15",
        dataset_version="candidate_v015",
        filtered_count=4,
        sample_count=20,
        reason="Candidate dataset blocked by NAT15 hardening test.",
        model_version="rd_feature_ae_gated_v001_bootstrap",
    )

    rows = list_incidents(
        incident_type="unsafe_train_candidate_blocked",
        scenario_id="scenario_nat15",
    )

    assert len(rows) == 1
    assert rows[0]["incident_id"] == incident["incident_id"]

    validated = Incident.model_validate(rows[0])

    assert validated.incident_type == "unsafe_train_candidate_blocked"
    assert validated.severity == "medium"
    assert validated.scenario_id == "scenario_nat15"
    assert validated.metadata["filtered_count"] == 4
