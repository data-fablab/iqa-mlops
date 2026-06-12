from __future__ import annotations

import pytest
from fastapi import HTTPException

from iqa.api.main import PredictRequest, app, health, model_version, predict


def test_api_app_exposes_phase_1_contract_routes() -> None:
    route_paths = {route.path for route in app.routes}

    assert "/health" in route_paths
    assert "/model/version" in route_paths
    assert "/predict" in route_paths


def test_health() -> None:
    assert health() == {"status": "ok", "service": "iqa-api"}


def test_model_version_reads_manifest_skeletons() -> None:
    response = model_version()

    assert response["roi_segmenter"]["model_type"] == "functional_unet_resnet18_det1_context2b"
    assert response["feature_ae"]["model_type"] == "reverse_distill_resnet18_dual_context_gated"


def test_predict_is_explicit_placeholder() -> None:
    request = PredictRequest(piece_event_id="piece-1", scenario_id="demo", image_uri="s3://bucket/key.jpg")

    with pytest.raises(HTTPException) as exc:
        predict(request)

    assert exc.value.status_code == 501
