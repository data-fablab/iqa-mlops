"""NAT03 scenario_id contract tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from iqa.api.main import model_version, reload_model
from iqa.api.schemas import PieceEventPredictRequest, PredictRequest, ReloadModelRequest


def test_predict_request_requires_scenario_id() -> None:
    with pytest.raises(ValidationError):
        PredictRequest(piece_event_id="piece_nat03_001", image_uri="s3://bucket/key.jpg")


def test_piece_event_predict_request_requires_scenario_id() -> None:
    with pytest.raises(ValidationError):
        PieceEventPredictRequest(image_uri="s3://bucket/key.jpg")


def test_reload_model_request_requires_scenario_id() -> None:
    with pytest.raises(ValidationError):
        ReloadModelRequest()


def test_model_version_is_scoped_by_scenario_id() -> None:
    response = model_version(scenario_id="scenario_nat03")

    assert response["scenario_id"] == "scenario_nat03"
    assert response["registered_model_name"]
    assert response["source_of_truth"] == "mlflow_registry"
    assert "roi_segmenter" in response
    assert "feature_ae" in response


def test_reload_model_response_is_scoped_by_scenario_id(monkeypatch) -> None:
    monkeypatch.setenv("IQA_ADMIN_TOKEN", "secret")

    response = reload_model(
        ReloadModelRequest(scenario_id="scenario_nat03"),
        x_iqa_admin_token="secret",
    )

    assert response["accepted"] is True
    assert response["audit"]["scenario_id"] == "scenario_nat03"
    assert response["target"]["scenario_id"] == "scenario_nat03"
