from __future__ import annotations

import pytest
from fastapi import HTTPException

from iqa.api import main as api
from iqa.api.schemas import FeedbackRequest, PredictRequest, ReloadModelRequest


class RecordingRepository:
    def __init__(self) -> None:
        self.predictions: dict[str, dict] = {}
        self.feedbacks: dict[str, dict] = {}
        self.display_feedbacks: dict[str, dict] = {}
        self.closed_predictions: list[tuple[str, str]] = []
        self.admin_reload_events: list[dict] = []

    def save_piece_event(self, piece_event_id: str, record: dict) -> None:
        pass

    def get_piece_event(self, piece_event_id: str) -> dict | None:
        return None

    def save_prediction(self, prediction_id: str, record: dict) -> None:
        self.predictions[prediction_id] = dict(record)

    def get_prediction(self, prediction_id: str) -> dict | None:
        return self.predictions.get(prediction_id)

    def list_predictions(self) -> list[dict]:
        return list(self.predictions.values())

    def save_feedback(self, prediction_id: str, record: dict) -> None:
        self.feedbacks[prediction_id] = dict(record)

    def get_feedback(self, prediction_id: str) -> dict | None:
        return self.feedbacks.get(prediction_id)

    def save_display_feedback(self, prediction_id: str, record: dict) -> None:
        self.display_feedbacks[prediction_id] = dict(record)

    def get_display_feedback(self, prediction_id: str) -> dict | None:
        return self.display_feedbacks.get(prediction_id)

    def mark_feedback_closed(self, prediction_id: str, closed_at: str) -> None:
        self.closed_predictions.append((prediction_id, closed_at))

    def save_admin_reload_event(self, record: dict) -> None:
        self.admin_reload_events.append(dict(record))

    def list_admin_reload_events(self) -> list[dict]:
        return list(self.admin_reload_events)


class FailingRepository(RecordingRepository):
    def save_prediction(self, prediction_id: str, record: dict) -> None:
        raise RuntimeError("database unavailable")


@pytest.fixture(autouse=True)
def _reset_api_state(monkeypatch: pytest.MonkeyPatch) -> None:
    api.PREDICTION_STORE.clear()
    api.FEEDBACK_STORE.clear()
    api.DISPLAY_FEEDBACK_STORE.clear()
    api.ADMIN_RELOAD_LOG.clear()
    api.METADATA_WRITE_THROUGH.reset()
    monkeypatch.delenv("IQA_METADATA_BACKEND", raising=False)
    monkeypatch.delenv("IQA_METADATA_DB_URL", raising=False)
    yield
    api.PREDICTION_STORE.clear()
    api.FEEDBACK_STORE.clear()
    api.DISPLAY_FEEDBACK_STORE.clear()
    api.ADMIN_RELOAD_LOG.clear()
    api.METADATA_WRITE_THROUGH.reset()


def test_memory_backend_does_not_create_metadata_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_called():
        raise AssertionError("PostgreSQL repository should not be created in memory mode")

    monkeypatch.setattr(api, "create_metadata_repository", fail_if_called)

    response = api.predict(
        PredictRequest(piece_event_id="piece_mem_001", scenario_id="demo", image_uri="s3://bucket/key.jpg")
    )

    assert response["prediction"]["prediction_id"] in api.PREDICTION_STORE


def test_postgres_backend_persists_predict_feedback_and_reload(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = RecordingRepository()
    monkeypatch.setenv("IQA_METADATA_BACKEND", "postgres")
    monkeypatch.setenv("IQA_ADMIN_TOKEN", "secret")
    monkeypatch.setattr(api, "create_metadata_repository", lambda: repo)

    prediction_response = api.predict(
        PredictRequest(piece_event_id="piece_pg_001", scenario_id="demo", image_uri="s3://bucket/key.jpg")
    )
    prediction_id = prediction_response["prediction"]["prediction_id"]

    api.feedback(
        FeedbackRequest(
            prediction_id=prediction_id,
            piece_event_id="piece_pg_001",
            scenario_id="demo",
            feedback_source="human_sophie",
        )
    )
    api.feedback(
        FeedbackRequest(
            prediction_id=prediction_id,
            piece_event_id="piece_pg_001",
            scenario_id="demo",
            feedback_source="oracle_gt",
            gt_mask_has_defect=True,
        )
    )
    api.reload_model(ReloadModelRequest(scenario_id="demo"), x_iqa_admin_token="secret")

    assert repo.predictions[prediction_id]["piece_event_id"] == "piece_pg_001"
    assert repo.display_feedbacks[prediction_id]["feedback_source"] == "human_sophie"
    assert repo.feedbacks[prediction_id]["feedback_source"] == "oracle_gt"
    assert repo.closed_predictions[0][0] == prediction_id
    assert repo.admin_reload_events[0]["reload_status"] == "accepted"


def test_postgres_write_failure_returns_503_without_memory_prediction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IQA_METADATA_BACKEND", "postgres")
    monkeypatch.setattr(api, "create_metadata_repository", lambda: FailingRepository())

    with pytest.raises(HTTPException) as exc_info:
        api.predict(PredictRequest(piece_event_id="piece_fail_001", scenario_id="demo", image_uri="s3://bucket/key.jpg"))

    assert exc_info.value.status_code == 503
    assert api.PREDICTION_STORE == {}


def test_postgres_backend_without_url_returns_503_without_memory_prediction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IQA_METADATA_BACKEND", "postgres")
    monkeypatch.delenv("IQA_METADATA_DB_URL", raising=False)

    with pytest.raises(HTTPException) as exc_info:
        api.predict(PredictRequest(piece_event_id="piece_no_url", scenario_id="demo", image_uri="s3://bucket/key.jpg"))

    assert exc_info.value.status_code == 503
    assert api.PREDICTION_STORE == {}
