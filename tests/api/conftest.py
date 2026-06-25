from __future__ import annotations

import pytest

from iqa.api import main as api
from iqa.inference.contracts import InferenceRequest, InferenceResult


@pytest.fixture(autouse=True)
def fake_inference_service(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    if request.node.path.name == "test_inference_http_delegation.py":
        return

    def fake_call(request: InferenceRequest) -> InferenceResult:
        return InferenceResult(
            piece_event_id=request.piece_event_id,
            scenario_id=request.scenario_id,
            score=0.0,
            decision="Vert",
            heatmap_uri=None,
            roi_status="ok",
            roi_model_version="roi_segmenter_v001_fixed",
            feature_ae_version="rd_feature_ae_gated_v001_bootstrap",
        )

    monkeypatch.setattr(
        api,
        "_call_inference_service",
        fake_call,
        raising=False,
    )
