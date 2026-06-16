"""NAT06 feedback poisoning control contracts."""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from iqa.api.main import (
    AI_SECURITY_METRICS,
    DISPLAY_FEEDBACK_STORE,
    FEEDBACK_STORE,
    PREDICTION_STORE,
    feedback,
    metrics,
    predict,
)
from iqa.api.schemas import FeedbackRequest, PredictRequest


@pytest.fixture(autouse=True)
def _reset_state() -> None:
    PREDICTION_STORE.clear()
    FEEDBACK_STORE.clear()
    DISPLAY_FEEDBACK_STORE.clear()
    for key in AI_SECURITY_METRICS:
        AI_SECURITY_METRICS[key] = 0
    yield
    PREDICTION_STORE.clear()
    FEEDBACK_STORE.clear()
    DISPLAY_FEEDBACK_STORE.clear()


def _create_prediction(piece_event_id: str = "piece_nat06", scenario_id: str = "scenario_nat06") -> str:
    response = predict(
        PredictRequest(
            piece_event_id=piece_event_id,
            scenario_id=scenario_id,
            image_uri="s3://iqa/raw/piece_nat06.png",
            sha256="6" * 64,
            lot_id="lot_nat06",
            dataset_version="casting_v006",
        )
    )
    return response["prediction"]["prediction_id"]


def test_feedback_poisoning_blocks_unknown_prediction_id() -> None:
    with pytest.raises(HTTPException) as exc_info:
        feedback(
            FeedbackRequest(
                prediction_id="pred_unknown",
                piece_event_id="piece_nat06_unknown",
                scenario_id="scenario_nat06",
                gt_mask_has_defect=False,
            )
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Unknown prediction_id."
    body = metrics()
    assert "iqa_invalid_feedback_total 1" in body
    assert "iqa_ai_security_incident_total 1" in body


def test_feedback_poisoning_blocks_piece_event_mismatch() -> None:
    prediction_id = _create_prediction(piece_event_id="piece_nat06_valid")

    with pytest.raises(HTTPException) as exc_info:
        feedback(
            FeedbackRequest(
                prediction_id=prediction_id,
                piece_event_id="piece_nat06_attacker",
                scenario_id="scenario_nat06",
                gt_mask_has_defect=False,
            )
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "prediction_id does not match piece_event_id."
    body = metrics()
    assert "iqa_feedback_conflict_total 1" in body
    assert "iqa_ai_security_incident_total 1" in body


def test_feedback_poisoning_blocks_scenario_mismatch() -> None:
    prediction_id = _create_prediction(piece_event_id="piece_nat06_valid", scenario_id="scenario_nat06_a")

    with pytest.raises(HTTPException) as exc_info:
        feedback(
            FeedbackRequest(
                prediction_id=prediction_id,
                piece_event_id="piece_nat06_valid",
                scenario_id="scenario_nat06_b",
                gt_mask_has_defect=False,
            )
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "prediction_id does not match scenario_id."
    body = metrics()
    assert "iqa_feedback_conflict_total 1" in body
    assert "iqa_ai_security_incident_total 1" in body


def test_feedback_poisoning_blocks_replay_after_closed_feedback() -> None:
    prediction_id = _create_prediction(piece_event_id="piece_nat06_replay")

    first = feedback(
        FeedbackRequest(
            prediction_id=prediction_id,
            piece_event_id="piece_nat06_replay",
            scenario_id="scenario_nat06",
            gt_mask_has_defect=False,
        )
    )

    assert first["accepted"] is True
    assert first["feedback_closed"] is True

    with pytest.raises(HTTPException) as exc_info:
        feedback(
            FeedbackRequest(
                prediction_id=prediction_id,
                piece_event_id="piece_nat06_replay",
                scenario_id="scenario_nat06",
                gt_mask_has_defect=True,
            )
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Prediction already has a closed feedback."
    body = metrics()
    assert "iqa_invalid_feedback_total 1" in body
    assert "iqa_ai_security_incident_total 1" in body


def test_feedback_poisoning_rejects_invalid_source_and_status_contracts() -> None:
    with pytest.raises(ValidationError):
        FeedbackRequest(
            prediction_id="pred_nat06",
            piece_event_id="piece_nat06",
            scenario_id="scenario_nat06",
            feedback_source="attacker_source",
        )

    with pytest.raises(ValidationError):
        FeedbackRequest(
            prediction_id="pred_nat06",
            piece_event_id="piece_nat06",
            scenario_id="scenario_nat06",
            feedback_status="poison_train",
        )


def test_human_sophie_feedback_is_blocked_from_train_candidates() -> None:
    prediction_id = _create_prediction(piece_event_id="piece_nat06_human")

    response = feedback(
        FeedbackRequest(
            prediction_id=prediction_id,
            piece_event_id="piece_nat06_human",
            scenario_id="scenario_nat06",
            feedback_source="human_sophie",
        )
    )

    assert response["accepted"] is True
    assert response["feedback_closed"] is False
    assert response["display_decision_source"] == "human_sophie"
    assert response["train_eligibility_source"] == "oracle_gt"
    assert response["eligible_for_train"] is False

    body = metrics()
    assert "iqa_unsafe_train_blocked_total 1" in body
