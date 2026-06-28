"""Tests for atomic inference runtime reload."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from iqa.inference import service
from iqa.inference.model_loader import LoadedModel
from iqa.models.feature_ae.reference import (
    REFERENCE_FEATURE_AE_CONTRACT,
)


def _bootstrap_runtime() -> (
    service.ActiveInferenceRuntime
):
    return service.ActiveInferenceRuntime(
        scenario_id=(
            "production_replay_natural"
        ),
        feature_ae_version=(
            "rd_feature_ae_gated_v001_bootstrap"
        ),
        roi_model_version=(
            "roi_segmenter_v001_fixed"
        ),
    )


def test_reload_swaps_runtime_only_after_success(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")

    loaded = LoadedModel(
        scenario_id=(
            "production_replay_natural"
        ),
        registered_model_name=(
            "feature_ae__production_replay_natural"
        ),
        version="12",
        artifact_uri="models:/m-serving-test",
        model=object(),
        feature_ae_version="candidate_v12",
        checkpoint_path=checkpoint,
        decision_thresholds={
            "threshold_orange": 1.0,
            "threshold_red": 2.0,
        },
        reference_contract=(
            REFERENCE_FEATURE_AE_CONTRACT
        ),
        roi_model_version=(
            "roi_segmenter_v001_fixed"
        ),
        model_id="m-serving-test",
    )

    loader = Mock()
    loader.reload.return_value = loaded

    monkeypatch.setenv(
        "IQA_ADMIN_TOKEN",
        "secret",
    )
    monkeypatch.setattr(
        service,
        "_ACTIVE_RUNTIME",
        _bootstrap_runtime(),
    )
    monkeypatch.setattr(
        service,
        "ProdModelLoader",
        lambda *args, **kwargs: loader,
    )

    payload = service.reload_model(
        service.ReloadInferenceRequest(
            scenario_id=(
                "production_replay_natural"
            ),
            stage="prod",
        ),
        x_iqa_admin_token="secret",
    )

    assert (
        payload["previous"][
            "feature_ae_version"
        ]
        == "rd_feature_ae_gated_v001_bootstrap"
    )
    assert (
        payload["active"][
            "feature_ae_version"
        ]
        == "candidate_v12"
    )
    assert (
        service.model_version()[
            "feature_ae_version"
        ]
        == "candidate_v12"
    )


def test_failed_reload_keeps_previous_runtime(
    monkeypatch,
) -> None:
    loader = Mock()
    loader.reload.side_effect = ValueError(
        "invalid bundle"
    )

    initial = _bootstrap_runtime()

    monkeypatch.setenv(
        "IQA_ADMIN_TOKEN",
        "secret",
    )
    monkeypatch.setattr(
        service,
        "_ACTIVE_RUNTIME",
        initial,
    )
    monkeypatch.setattr(
        service,
        "ProdModelLoader",
        lambda *args, **kwargs: loader,
    )

    with pytest.raises(
        HTTPException,
    ) as error:
        service.reload_model(
            service.ReloadInferenceRequest(
                scenario_id=(
                    "production_replay_natural"
                ),
                stage="prod",
            ),
            x_iqa_admin_token="secret",
        )

    assert error.value.status_code == 503
    assert service._runtime_snapshot() == initial


def test_prediction_uses_one_runtime_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")

    runtime = service.ActiveInferenceRuntime(
        scenario_id=(
            "production_replay_natural"
        ),
        feature_ae_version="candidate_v12",
        roi_model_version=(
            "roi_segmenter_v001_fixed"
        ),
        checkpoint_path=checkpoint,
        decision_thresholds={
            "threshold_orange": 1.0,
            "threshold_red": 2.0,
        },
        reference_contract=(
            REFERENCE_FEATURE_AE_CONTRACT
        ),
    )
    monkeypatch.setattr(
        service,
        "_ACTIVE_RUNTIME",
        runtime,
    )

    result = Mock()
    result.result = Mock()
    pipeline = Mock(return_value=result)

    monkeypatch.setattr(
        service,
        "run_inference_pipeline",
        pipeline,
    )

    request = Mock()
    service._run_real_inference(request)

    pipeline.assert_called_once_with(
        request,
        device="cpu",
        roi_model_version=(
            "roi_segmenter_v001_fixed"
        ),
        feature_ae_version="candidate_v12",
        feature_checkpoint=checkpoint,
        decision_thresholds={
            "threshold_orange": 1.0,
            "threshold_red": 2.0,
        },
        feature_ae_reference_contract=(
            REFERENCE_FEATURE_AE_CONTRACT
        ),
    )
