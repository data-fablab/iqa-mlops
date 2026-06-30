from __future__ import annotations

import pytest
from fastapi import HTTPException

from iqa.api import main as api
from iqa.api.schemas import FeedbackRequest, PredictRequest, ReloadModelRequest
from iqa.metadata.repository import MemoryMetadataRepository


class RecordingRepository:
    def __init__(self) -> None:
        self.predictions: dict[str, dict] = {}
        self.feedbacks: dict[str, dict] = {}
        self.display_feedbacks: dict[str, dict] = {}
        self.closed_predictions: list[tuple[str, str]] = []
        self.admin_reload_events: list[dict] = []
        self.incidents: list[dict] = []

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

    def save_feedback_and_close_prediction(
        self,
        prediction_id: str,
        feedback_record: dict,
        closed_at: str,
    ) -> None:
        self.save_feedback(prediction_id, feedback_record)
        self.mark_feedback_closed(prediction_id, closed_at)

    def get_feedback(self, prediction_id: str) -> dict | None:
        return self.feedbacks.get(prediction_id)

    def save_display_feedback(self, prediction_id: str, record: dict) -> None:
        self.display_feedbacks[prediction_id] = dict(record)

    def get_display_feedback(self, prediction_id: str) -> dict | None:
        return self.display_feedbacks.get(prediction_id)

    def mark_feedback_closed(self, prediction_id: str, closed_at: str) -> None:
        self.closed_predictions.append((prediction_id, closed_at))
        if prediction_id in self.predictions:
            self.predictions[prediction_id]["feedback_closed"] = True

    def save_admin_reload_event(self, record: dict) -> None:
        self.admin_reload_events.append(dict(record))

    def list_admin_reload_events(self) -> list[dict]:
        return list(self.admin_reload_events)

    def save_incident_event(self, record: dict) -> None:
        self.incidents.append(dict(record))

    def list_incident_events(self) -> list[dict]:
        return list(self.incidents)


class FailingRepository(RecordingRepository):
    def save_prediction(self, prediction_id: str, record: dict) -> None:
        raise RuntimeError("database unavailable")


class FailingAdminReloadRepository(RecordingRepository):
    def save_admin_reload_event(self, record: dict) -> None:
        raise RuntimeError("database unavailable")


@pytest.fixture(autouse=True)
def _reset_api_state(monkeypatch: pytest.MonkeyPatch) -> None:
    api.METADATA_REPOSITORY.reset()
    for metric in api.AI_SECURITY_METRICS:
        api.AI_SECURITY_METRICS[metric] = 0
    monkeypatch.delenv("IQA_METADATA_BACKEND", raising=False)
    monkeypatch.delenv("IQA_METADATA_DB_URL", raising=False)
    monkeypatch.delenv("IQA_SERVICE_TOKEN", raising=False)
    monkeypatch.delenv("IQA_ADMIN_TOKEN", raising=False)
    yield
    api.METADATA_REPOSITORY.reset()
    for metric in api.AI_SECURITY_METRICS:
        api.AI_SECURITY_METRICS[metric] = 0


def test_memory_backend_persists_prediction_in_memory_adapter() -> None:
    response = api.predict(
        PredictRequest(piece_event_id="piece_mem_001", scenario_id="demo", image_uri="s3://bucket/key.jpg")
    )

    prediction_id = response["prediction"]["prediction_id"]
    repository = api.metadata_repository()

    assert isinstance(repository, MemoryMetadataRepository)
    assert repository.get_prediction(prediction_id)["piece_event_id"] == "piece_mem_001"


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


def test_postgres_write_failure_returns_503_without_persisting_prediction(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = FailingRepository()
    monkeypatch.setenv("IQA_METADATA_BACKEND", "postgres")
    monkeypatch.setattr(api, "create_metadata_repository", lambda: repo)

    with pytest.raises(HTTPException) as exc_info:
        api.predict(PredictRequest(piece_event_id="piece_fail_001", scenario_id="demo", image_uri="s3://bucket/key.jpg"))

    assert exc_info.value.status_code == 503
    assert repo.predictions == {}


def test_admin_reload_refusal_keeps_security_audit_when_postgres_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = FailingAdminReloadRepository()
    monkeypatch.setenv("IQA_METADATA_BACKEND", "postgres")
    monkeypatch.setenv("IQA_ADMIN_TOKEN", "secret")
    monkeypatch.setattr(api, "create_metadata_repository", lambda: repo)

    with pytest.raises(HTTPException) as exc_info:
        api.reload_model(ReloadModelRequest(scenario_id="demo"), x_iqa_admin_token="wrong")

    assert exc_info.value.status_code == 401
    assert api.AI_SECURITY_METRICS["reload_refused_total"] == 1
    # The durable reload-event write failed best-effort (no exception bubbled up),
    # while the security incident is still recorded through the same seam.
    assert repo.admin_reload_events == []
    assert repo.incidents[0]["incident_type"] == "reload_refused"


def test_postgres_backend_without_url_returns_503(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IQA_METADATA_BACKEND", "postgres")
    monkeypatch.delenv("IQA_METADATA_DB_URL", raising=False)

    with pytest.raises(HTTPException) as exc_info:
        api.predict(PredictRequest(piece_event_id="piece_no_url", scenario_id="demo", image_uri="s3://bucket/key.jpg"))

    assert exc_info.value.status_code == 503
