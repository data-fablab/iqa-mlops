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
    app,
    feedback,
    health,
    model_version,
    metrics,
    predict,
    predict_piece_event,
    reload_model,
    replay_scenarios,
)
from iqa.inference.service import health as inference_health
from iqa.inference.service import metrics as inference_metrics


def test_api_app_exposes_phase_1_contract_routes() -> None:
    route_paths = {route.path for route in app.routes}

    assert "/health" in route_paths
    assert "/model/version" in route_paths
    assert "/predict" in route_paths
    assert "/piece-events/{event_id}/predict" in route_paths
    assert "/feedback" in route_paths
    assert "/replay-scenarios" in route_paths
    assert "/admin/reload-model" in route_paths


def test_health() -> None:
    assert health() == {"status": "ok", "service": "iqa-api"}


def test_model_version_reads_manifest_skeletons() -> None:
    response = model_version()

    assert response["roi_segmenter"]["model_type"] == "functional_unet_resnet18_det1_context2b"
    assert response["feature_ae"]["model_type"] == "reverse_distill_resnet18_dual_context_gated"


def test_predict_is_explicit_placeholder() -> None:
    request = PredictRequest(piece_event_id="piece-1", scenario_id="demo", image_uri="s3://bucket/key.jpg")

    response = predict(request)

    assert response["delegated_to"] == "iqa-inference"
    assert response["prediction"]["decision"] == "Vert"


def test_piece_event_predict_uses_path_event_id() -> None:
    response = predict_piece_event(
        "piece-from-path",
        PieceEventPredictRequest(scenario_id="demo", image_uri="s3://bucket/key.jpg"),
    )

    assert response["prediction"]["piece_event_id"] == "piece-from-path"
    assert response["prediction"]["decision"] == "Vert"
    assert response["prediction"]["prediction_id"].startswith("pred_")
    assert response["prediction"]["audit_logged"] is True
    assert response["audit"]["audit_logged"] is True
    assert response["audit"]["piece_event_id"] == "piece-from-path"
    assert response["audit"]["scenario_id"] == "demo"
    assert response["audit"]["image_uri"] == "s3://bucket/key.jpg"
    assert response["audit"]["decision"] == "Vert"
    assert response["audit"]["audit_sink"] == "api_response_mvp"


def test_inference_service_health() -> None:
    assert inference_health() == {"status": "ok", "service": "iqa-inference"}


def test_inference_service_metrics() -> None:
    assert "iqa_inference_up 1" in inference_metrics()


def test_feedback_accepts_oracle_gt() -> None:
    prediction_response = predict(
        PredictRequest(piece_event_id="piece-feedback-ok", scenario_id="demo", image_uri="s3://bucket/key.jpg")
    )
    prediction_id = prediction_response["prediction"]["prediction_id"]

    response = feedback(
        FeedbackRequest(
            prediction_id=prediction_id,
            piece_event_id="piece-feedback-ok",
            scenario_id="demo",
            gt_mask_has_defect=True,
        )
    )

    assert response["accepted"] is True
    assert response["prediction_id"] == prediction_id
    assert response["feedback_closed"] is True
    assert response["feedback"]["feedback_source"] == "oracle_gt"
    assert response["feedback"]["verdict"] == "defective"


def test_feedback_accepts_human_sophie_for_display_only() -> None:
    prediction_response = predict(
        PredictRequest(piece_event_id="piece-human-sophie", scenario_id="demo", image_uri="s3://bucket/key.jpg")
    )
    prediction_id = prediction_response["prediction"]["prediction_id"]

    response = feedback(
        FeedbackRequest(
            prediction_id=prediction_id,
            piece_event_id="piece-human-sophie",
            scenario_id="demo",
            feedback_source="human_sophie",
        )
    )

    assert response["accepted"] is True
    assert response["prediction_id"] == prediction_id
    assert response["feedback_closed"] is False
    assert response["display_decision_source"] == "human_sophie"
    assert response["train_eligibility_source"] == "oracle_gt"
    assert response["eligible_for_train"] is False
    assert "oracle_gt" in response["reason"]


def test_oracle_gt_remains_sovereign_after_human_sophie_feedback() -> None:
    prediction_response = predict(
        PredictRequest(piece_event_id="piece-human-then-gt", scenario_id="demo", image_uri="s3://bucket/key.jpg")
    )
    prediction_id = prediction_response["prediction"]["prediction_id"]

    human_response = feedback(
        FeedbackRequest(
            prediction_id=prediction_id,
            piece_event_id="piece-human-then-gt",
            scenario_id="demo",
            feedback_source="human_sophie",
        )
    )

    assert human_response["accepted"] is True
    assert human_response["feedback_closed"] is False
    assert human_response["display_decision_source"] == "human_sophie"

    gt_response = feedback(
        FeedbackRequest(
            prediction_id=prediction_id,
            piece_event_id="piece-human-then-gt",
            scenario_id="demo",
            feedback_source="oracle_gt",
            gt_mask_has_defect=False,
        )
    )

    assert gt_response["accepted"] is True
    assert gt_response["feedback_closed"] is True
    assert gt_response["display_decision_source"] == "human_sophie"
    assert gt_response["train_eligibility_source"] == "oracle_gt"
    assert gt_response["eligible_for_train"] is True
    assert gt_response["feedback"]["verdict"] == "conforme"


def test_feedback_rejects_unknown_prediction_id() -> None:
    with pytest.raises(HTTPException) as exc_info:
        feedback(
            FeedbackRequest(
                prediction_id="pred_unknown",
                piece_event_id="piece-unknown",
                scenario_id="demo",
                gt_mask_has_defect=True,
            )
        )

    assert exc_info.value.status_code == 404


def test_feedback_rejects_closed_prediction() -> None:
    prediction_response = predict(
        PredictRequest(piece_event_id="piece-closed", scenario_id="demo", image_uri="s3://bucket/key.jpg")
    )
    prediction_id = prediction_response["prediction"]["prediction_id"]

    first_response = feedback(
        FeedbackRequest(
            prediction_id=prediction_id,
            piece_event_id="piece-closed",
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
                piece_event_id="piece-closed",
                scenario_id="demo",
                gt_mask_has_defect=True,
            )
        )

    assert exc_info.value.status_code == 409



def test_feedback_rejects_unknown_feedback_source_contract() -> None:
    with pytest.raises(ValidationError):
        FeedbackRequest(
            prediction_id="pred-contract-source",
            piece_event_id="piece-contract-source",
            scenario_id="demo",
            feedback_source="unknown_source",
        )


def test_feedback_rejects_forbidden_feedback_status_contract() -> None:
    with pytest.raises(ValidationError):
        FeedbackRequest(
            prediction_id="pred-contract-status",
            piece_event_id="piece-contract-status",
            scenario_id="demo",
            feedback_status="invalid_status",
        )


def test_feedback_rejects_prediction_piece_event_mismatch() -> None:
    prediction_response = predict(
        PredictRequest(piece_event_id="piece-valid", scenario_id="demo", image_uri="s3://bucket/key.jpg")
    )
    prediction_id = prediction_response["prediction"]["prediction_id"]

    with pytest.raises(HTTPException) as exc_info:
        feedback(
            FeedbackRequest(
                prediction_id=prediction_id,
                piece_event_id="piece-other",
                scenario_id="demo",
                gt_mask_has_defect=True,
            )
        )

    assert exc_info.value.status_code == 409


def test_feedback_rejects_prediction_scenario_mismatch() -> None:
    prediction_response = predict(
        PredictRequest(piece_event_id="piece-scenario", scenario_id="demo", image_uri="s3://bucket/key.jpg")
    )
    prediction_id = prediction_response["prediction"]["prediction_id"]

    with pytest.raises(HTTPException) as exc_info:
        feedback(
            FeedbackRequest(
                prediction_id=prediction_id,
                piece_event_id="piece-scenario",
                scenario_id="other-scenario",
                gt_mask_has_defect=True,
            )
        )

    assert exc_info.value.status_code == 409



def test_metrics_exposes_ai_security_counters() -> None:
    body = metrics()

    assert "iqa_feedback_conflict_total" in body
    assert "iqa_ai_security_incident_total" in body
    assert "iqa_unsafe_train_blocked_total" in body
    assert "iqa_invalid_feedback_total" in body
    assert "iqa_reload_refused_total" in body


def test_metrics_count_feedback_conflict_and_train_block() -> None:
    for key in AI_SECURITY_METRICS:
        AI_SECURITY_METRICS[key] = 0

    prediction_response = predict(
        PredictRequest(piece_event_id="piece-metrics", scenario_id="demo", image_uri="s3://bucket/key.jpg")
    )
    prediction_id = prediction_response["prediction"]["prediction_id"]

    with pytest.raises(HTTPException):
        feedback(
            FeedbackRequest(
                prediction_id=prediction_id,
                piece_event_id="piece-metrics-other",
                scenario_id="demo",
                gt_mask_has_defect=True,
            )
        )

    prediction_response = predict(
        PredictRequest(piece_event_id="piece-metrics-human", scenario_id="demo", image_uri="s3://bucket/key.jpg")
    )
    prediction_id = prediction_response["prediction"]["prediction_id"]

    feedback(
        FeedbackRequest(
            prediction_id=prediction_id,
            piece_event_id="piece-metrics-human",
            scenario_id="demo",
            feedback_source="human_sophie",
        )
    )

    body = metrics()

    assert "iqa_feedback_conflict_total 1" in body
    assert "iqa_ai_security_incident_total 1" in body
    assert "iqa_unsafe_train_blocked_total 1" in body


def test_admin_reload_fails_when_admin_token_is_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    ADMIN_RELOAD_LOG.clear()
    monkeypatch.delenv("IQA_ADMIN_TOKEN", raising=False)

    with pytest.raises(HTTPException) as exc_info:
        reload_model(ReloadModelRequest(scenario_id="demo"), x_iqa_admin_token="secret")

    assert exc_info.value.status_code == 503
    assert ADMIN_RELOAD_LOG[-1]["reload_status"] == "refused"
    assert ADMIN_RELOAD_LOG[-1]["accepted"] is False
    assert ADMIN_RELOAD_LOG[-1]["reason"] == "IQA_ADMIN_TOKEN is not configured."


def test_admin_reload_requires_token_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    ADMIN_RELOAD_LOG.clear()
    monkeypatch.setenv("IQA_ADMIN_TOKEN", "secret")

    with pytest.raises(HTTPException) as exc_info:
        reload_model(ReloadModelRequest(scenario_id="demo"), x_iqa_admin_token="bad")

    assert exc_info.value.status_code == 401
    assert ADMIN_RELOAD_LOG[-1]["reload_status"] == "refused"
    assert ADMIN_RELOAD_LOG[-1]["accepted"] is False
    assert ADMIN_RELOAD_LOG[-1]["reason"] == "Missing or invalid IQA_ADMIN_TOKEN."

    response = reload_model(ReloadModelRequest(scenario_id="demo"), x_iqa_admin_token="secret")

    assert response["accepted"] is True
    assert response["reload_status"] == "accepted"
    assert response["source_of_truth"] == "mlflow_registry"
    assert response["audit_logged"] is True
    assert response["audit"]["reload_status"] == "accepted"
    assert response["audit"]["accepted"] is True
    assert response["audit"]["scenario_id"] == "demo"
    assert ADMIN_RELOAD_LOG[-1]["reload_status"] == "accepted"


def test_replay_scenarios_endpoint() -> None:
    scenario_ids = {scenario["scenario_id"] for scenario in replay_scenarios()}

    assert {"production_replay_natural", "drift_domain_extension"} <= scenario_ids
