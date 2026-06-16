"""NAT14 API incident creation contracts."""

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
    feedback,
    list_incidents,
    predict,
    record_dataset_blocked_incident,
    reload_model,
)
from iqa.api.schemas import FeedbackRequest, FeedbackStatus, PredictRequest, ReloadModelRequest


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch: pytest.MonkeyPatch) -> None:
    PREDICTION_STORE.clear()
    FEEDBACK_STORE.clear()
    DISPLAY_FEEDBACK_STORE.clear()
    INCIDENT_STORE.clear()
    ADMIN_RELOAD_LOG.clear()
    for key in AI_SECURITY_METRICS:
        AI_SECURITY_METRICS[key] = 0
    monkeypatch.delenv("IQA_ADMIN_TOKEN", raising=False)
    monkeypatch.delenv("IQA_SERVICE_TOKEN", raising=False)
    yield
    PREDICTION_STORE.clear()
    FEEDBACK_STORE.clear()
    DISPLAY_FEEDBACK_STORE.clear()
    INCIDENT_STORE.clear()
    ADMIN_RELOAD_LOG.clear()


def _create_prediction(piece_event_id: str = "piece_nat14", scenario_id: str = "scenario_nat14") -> str:
    response = predict(
        PredictRequest(
            piece_event_id=piece_event_id,
            scenario_id=scenario_id,
            image_uri=f"s3://iqa/raw/{piece_event_id}.png",
            sha256="4" * 64,
            lot_id="lot_nat14",
            source_class="Casting_class1",
            dataset_version="casting_v014",
        )
    )
    return response["prediction"]["prediction_id"]


def test_nat14_creates_feedback_conflict_incident() -> None:
    prediction_id = _create_prediction(piece_event_id="piece_nat14_conflict")

    with pytest.raises(HTTPException) as exc_info:
        feedback(
            FeedbackRequest(
                prediction_id=prediction_id,
                piece_event_id="piece_nat14_attacker",
                scenario_id="scenario_nat14",
            )
        )

    assert exc_info.value.status_code == 409
    incidents = list_incidents()
    assert len(incidents) == 1
    assert incidents[0]["incident_type"] == "feedback_conflict"
    assert incidents[0]["severity"] == "medium"
    assert incidents[0]["prediction_id"] == prediction_id
    assert incidents[0]["metadata"]["expected_piece_event_id"] == "piece_nat14_conflict"
    assert incidents[0]["metadata"]["received_piece_event_id"] == "piece_nat14_attacker"


def test_nat14_creates_false_negative_incident() -> None:
    prediction_id = _create_prediction(piece_event_id="piece_nat14_fn")
    PREDICTION_STORE[prediction_id]["decision"] = "Vert"

    feedback(
        FeedbackRequest(
            prediction_id=prediction_id,
            piece_event_id="piece_nat14_fn",
            scenario_id="scenario_nat14",
            feedback_source="oracle_gt",
            gt_mask_has_defect=True,
        )
    )

    incidents = list_incidents(incident_type="false_negative")
    assert len(incidents) == 1
    assert incidents[0]["severity"] == "high"
    assert incidents[0]["prediction_id"] == prediction_id
    assert incidents[0]["metadata"]["divergence"] == "faux_negatif"
    assert incidents[0]["metadata"]["oracle_verdict"] == "defective"


def test_nat14_creates_roi_fail_incident() -> None:
    prediction_id = _create_prediction(piece_event_id="piece_nat14_roi_fail")

    feedback(
        FeedbackRequest(
            prediction_id=prediction_id,
            piece_event_id="piece_nat14_roi_fail",
            scenario_id="scenario_nat14",
            feedback_source="oracle_gt",
            feedback_status=FeedbackStatus.roi_fail,
            gt_mask_has_defect=False,
        )
    )

    incidents = list_incidents(incident_type="roi_fail")
    assert len(incidents) == 1
    assert incidents[0]["severity"] == "high"
    assert incidents[0]["prediction_id"] == prediction_id
    assert incidents[0]["metadata"]["train_block_reason"] == "roi_fail"


def test_nat14_creates_reload_refused_incident() -> None:
    with pytest.raises(HTTPException) as exc_info:
        reload_model(ReloadModelRequest(scenario_id="scenario_nat14"), x_iqa_admin_token="secret")

    assert exc_info.value.status_code == 503
    incidents = list_incidents(incident_type="reload_refused")
    assert len(incidents) == 1
    assert incidents[0]["severity"] == "high"
    assert incidents[0]["scenario_id"] == "scenario_nat14"
    assert incidents[0]["metadata"]["reload_event_id"] == ADMIN_RELOAD_LOG[-1]["reload_event_id"]


def test_nat14_creates_dataset_blocked_incident() -> None:
    incident = record_dataset_blocked_incident(
        scenario_id="scenario_nat14",
        dataset_version="candidate_v014",
        filtered_count=12,
        sample_count=40,
        reason="Candidate dataset blocked by ROI or GT safety filtering.",
        model_version="rd_feature_ae_gated_v001_bootstrap",
    )

    assert incident["incident_type"] == "unsafe_train_candidate_blocked"
    assert incident["severity"] == "medium"
    assert incident["scenario_id"] == "scenario_nat14"
    assert incident["model_version"] == "rd_feature_ae_gated_v001_bootstrap"
    assert incident["metadata"]["dataset_version"] == "candidate_v014"
    assert incident["metadata"]["filtered_count"] == 12
    assert list_incidents(scenario_id="scenario_nat14")[0]["incident_id"] == incident["incident_id"]
