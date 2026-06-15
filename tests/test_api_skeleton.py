from __future__ import annotations

import pytest
from fastapi import HTTPException

from iqa.api.main import (
    FeedbackRequest,
    PieceEventPredictRequest,
    PredictRequest,
    ReloadModelRequest,
    app,
    feedback,
    health,
    model_version,
    predict,
    predict_piece_event,
    reload_model,
    replay_scenarios,
)
from iqa.inference.service import health as inference_health


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
    assert response["prediction"]["statut"] == "Vert"


def test_piece_event_predict_uses_path_event_id() -> None:
    response = predict_piece_event(
        "piece-from-path",
        PieceEventPredictRequest(scenario_id="demo", image_uri="s3://bucket/key.jpg"),
    )

    assert response["prediction"]["piece_event_id"] == "piece-from-path"


def test_inference_service_health() -> None:
    assert inference_health() == {"status": "ok", "service": "iqa-inference"}


def test_feedback_accepts_oracle_gt() -> None:
    response = feedback(FeedbackRequest(piece_event_id="piece-1", scenario_id="demo", gt_mask_has_defect=True))

    assert response["accepted"] is True
    assert response["feedback"]["feedback_source"] == "oracle_gt"
    assert response["feedback"]["verdict"] == "defective"


def test_feedback_rejects_human_sophie_for_mvp() -> None:
    response = feedback(FeedbackRequest(piece_event_id="piece-1", scenario_id="demo", feedback_source="human_sophie"))

    assert response["accepted"] is False
    assert "oracle_gt" in response["reason"]


def test_admin_reload_requires_token_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IQA_ADMIN_TOKEN", "secret")

    with pytest.raises(HTTPException) as exc_info:
        reload_model(ReloadModelRequest(scenario_id="demo"), x_iqa_admin_token="bad")

    assert exc_info.value.status_code == 401
    response = reload_model(ReloadModelRequest(scenario_id="demo"), x_iqa_admin_token="secret")
    assert response["source_of_truth"] == "mlflow_registry"


def test_replay_scenarios_endpoint() -> None:
    scenario_ids = {scenario["scenario_id"] for scenario in replay_scenarios()}

    assert {"production_replay_natural", "drift_domain_extension"} <= scenario_ids
