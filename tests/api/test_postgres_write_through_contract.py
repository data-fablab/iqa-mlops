from __future__ import annotations

import os

import pytest

from iqa.api import main as api
from iqa.api.schemas import FeedbackRequest, PredictRequest, ReloadModelRequest
from iqa.metadata.postgres import PostgresMetadataRepository, initialize_metadata_db


@pytest.fixture()
def postgres_api_repo(monkeypatch: pytest.MonkeyPatch) -> PostgresMetadataRepository:
    db_url = os.getenv("IQA_METADATA_DB_URL")
    if not db_url:
        pytest.skip("IQA_METADATA_DB_URL is not set.")

    initialize_metadata_db(db_url)
    api.PREDICTION_STORE.clear()
    api.FEEDBACK_STORE.clear()
    api.DISPLAY_FEEDBACK_STORE.clear()
    api.ADMIN_RELOAD_LOG.clear()
    api.METADATA_WRITE_THROUGH.reset()
    monkeypatch.setenv("IQA_METADATA_BACKEND", "postgres")
    monkeypatch.delenv("IQA_SERVICE_TOKEN", raising=False)
    monkeypatch.setenv("IQA_ADMIN_TOKEN", "secret")
    yield PostgresMetadataRepository(db_url)
    api.PREDICTION_STORE.clear()
    api.FEEDBACK_STORE.clear()
    api.DISPLAY_FEEDBACK_STORE.clear()
    api.ADMIN_RELOAD_LOG.clear()
    api.METADATA_WRITE_THROUGH.reset()


@pytest.mark.postgres_contract
def test_api_write_through_persists_prediction_feedback_and_reload(
    postgres_api_repo: PostgresMetadataRepository,
) -> None:
    response = api.predict(
        PredictRequest(
            piece_event_id="piece_api_pg_001",
            scenario_id="scenario_api_pg",
            image_uri="s3://iqa/raw/piece_api_pg_001.png",
            sha256="d" * 64,
            lot_id="lot_api_pg",
            dataset_version="casting_api_pg_v001",
        )
    )
    prediction_id = response["prediction"]["prediction_id"]

    api.feedback(
        FeedbackRequest(
            prediction_id=prediction_id,
            piece_event_id="piece_api_pg_001",
            scenario_id="scenario_api_pg",
            feedback_source="human_sophie",
        )
    )
    api.feedback(
        FeedbackRequest(
            prediction_id=prediction_id,
            piece_event_id="piece_api_pg_001",
            scenario_id="scenario_api_pg",
            feedback_source="oracle_gt",
            gt_mask_has_defect=False,
        )
    )
    reload_response = api.reload_model(
        ReloadModelRequest(scenario_id="scenario_api_pg"),
        x_iqa_admin_token="secret",
    )

    saved_prediction = postgres_api_repo.get_prediction(prediction_id)
    saved_display = postgres_api_repo.get_display_feedback(prediction_id)
    saved_feedback = postgres_api_repo.get_feedback(prediction_id)
    reload_event_ids = {event["reload_event_id"] for event in postgres_api_repo.list_admin_reload_events()}

    assert saved_prediction["piece_event_id"] == "piece_api_pg_001"
    assert saved_prediction["feedback_closed"] is True
    assert saved_display["feedback_source"] == "human_sophie"
    assert saved_feedback["feedback_source"] == "oracle_gt"
    assert reload_response["audit"]["reload_event_id"] in reload_event_ids
