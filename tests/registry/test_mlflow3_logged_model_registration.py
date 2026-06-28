"""Tests for MLflow 3 LoggedModel registration."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from iqa.registry.mlflow_registry import (
    register_logged_feature_ae_model,
)


def _artifact(path: str, *, is_dir: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        path=path,
        is_dir=is_dir,
    )


def _client(*, complete: bool = True) -> Mock:
    client = Mock()
    client.get_logged_model.return_value = SimpleNamespace(model_id="m-serving-test")

    root = [
        _artifact("MLmodel"),
        _artifact("artifacts", is_dir=True),
    ]
    bundle = [
        _artifact("artifacts/checkpoint.pt"),
        _artifact("artifacts/model_manifest.json"),
        _artifact("artifacts/score_contract.json"),
    ]

    if not complete:
        bundle = [
            item for item in bundle if item.path != "artifacts/model_manifest.json"
        ]

    def list_artifacts(
        model_id: str,
        path: str | None = None,
    ) -> list[SimpleNamespace]:
        assert model_id == "m-serving-test"
        return root if path is None else bundle

    client.list_logged_model_artifacts.side_effect = list_artifacts
    client.create_model_version.return_value = SimpleNamespace(version="12")
    return client


def test_register_logged_model_uses_mlflow3_source() -> None:
    client = _client()

    with patch(
        "iqa.registry.mlflow_registry.mlflow.tracking.MlflowClient",
        return_value=client,
    ):
        result = register_logged_feature_ae_model(
            model_uri="models:/m-serving-test",
            run_id="run-123",
            scenario_id="production_replay_natural",
            stage="prod",
        )

    assert result["version"] == "12"
    assert result["model_id"] == "m-serving-test"
    assert result["source_of_truth"] == ("mlflow_logged_model")

    client.create_model_version.assert_called_once_with(
        name="feature_ae__production_replay_natural",
        source="models:/m-serving-test",
        run_id="run-123",
        model_id="m-serving-test",
    )
    client.set_registered_model_alias.assert_called_once_with(
        name="feature_ae__production_replay_natural",
        alias="prod",
        version="12",
    )


def test_register_logged_model_rejects_incomplete_bundle() -> None:
    client = _client(complete=False)

    with (
        patch(
            "iqa.registry.mlflow_registry.mlflow.tracking.MlflowClient",
            return_value=client,
        ),
        pytest.raises(
            FileNotFoundError,
            match="model_manifest.json",
        ),
    ):
        register_logged_feature_ae_model(
            model_uri="models:/m-serving-test",
            run_id="run-123",
            scenario_id="production_replay_natural",
            stage="prod",
        )

    client.create_model_version.assert_not_called()
