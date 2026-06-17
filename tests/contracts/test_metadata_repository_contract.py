"""Contract tests for the IQA metadata repository foundation."""

import pytest

from iqa.metadata.repository import MemoryMetadataRepository, metadata_db_url


def test_piece_event_is_saved_and_returned_as_copy():
    repo = MemoryMetadataRepository()

    record = {
        "piece_event_id": "piece_001",
        "source_event_id": "source_piece_001",
        "scenario_id": "demo",
        "payload": {"manifest_id": "casting_piece_events_v001"},
    }

    repo.save_piece_event("piece_001", record)
    record["scenario_id"] = "changed"

    saved = repo.get_piece_event("piece_001")
    assert saved is not None
    assert saved["scenario_id"] == "demo"
    assert saved["payload"]["manifest_id"] == "casting_piece_events_v001"

    saved["payload"]["manifest_id"] = "changed"

    saved_again = repo.get_piece_event("piece_001")
    assert saved_again is not None
    assert saved_again["payload"]["manifest_id"] == "casting_piece_events_v001"


def test_prediction_is_saved_listed_and_returned_as_copy():
    repo = MemoryMetadataRepository()

    record = {
        "prediction_id": "pred_001",
        "piece_event_id": "piece_001",
        "scenario_id": "demo",
        "feedback_closed": False,
        "audit": {"model_version": "feature_ae_v1"},
    }

    repo.save_prediction("pred_001", record)

    record["piece_event_id"] = "changed"

    saved = repo.get_prediction("pred_001")
    assert saved is not None
    assert saved["piece_event_id"] == "piece_001"
    assert saved["audit"]["model_version"] == "feature_ae_v1"

    saved["audit"]["model_version"] = "changed"

    saved_again = repo.get_prediction("pred_001")
    assert saved_again is not None
    assert saved_again["audit"]["model_version"] == "feature_ae_v1"

    rows = repo.list_predictions()
    assert len(rows) == 1
    assert rows[0]["prediction_id"] == "pred_001"


def test_oracle_feedback_and_display_feedback_are_kept_separate():
    repo = MemoryMetadataRepository()

    oracle_feedback = {
        "prediction_id": "pred_001",
        "feedback_source": "oracle_gt",
        "eligible_for_train": True,
    }

    display_feedback = {
        "prediction_id": "pred_001",
        "feedback_source": "human_sophie",
        "eligible_for_train": False,
        "display_decision": "accepted_for_display",
    }

    repo.save_feedback("pred_001", oracle_feedback)
    repo.save_display_feedback("pred_001", display_feedback)

    assert repo.get_feedback("pred_001")["feedback_source"] == "oracle_gt"
    assert repo.get_display_feedback("pred_001")["feedback_source"] == "human_sophie"
    assert repo.get_display_feedback("pred_001")["eligible_for_train"] is False


def test_mark_feedback_closed_updates_prediction_state():
    repo = MemoryMetadataRepository()

    repo.save_prediction(
        "pred_001",
        {
            "prediction_id": "pred_001",
            "piece_event_id": "piece_001",
            "scenario_id": "demo",
            "feedback_closed": False,
        },
    )

    repo.mark_feedback_closed("pred_001", "2026-06-16T15:00:00Z")

    saved = repo.get_prediction("pred_001")
    assert saved is not None
    assert saved["feedback_closed"] is True
    assert saved["feedback_closed_at"] == "2026-06-16T15:00:00Z"


def test_mark_feedback_closed_unknown_prediction_raises_key_error():
    repo = MemoryMetadataRepository()

    with pytest.raises(KeyError):
        repo.mark_feedback_closed("pred_unknown", "2026-06-16T15:00:00Z")


def test_admin_reload_events_are_appended_and_returned_as_copy():
    repo = MemoryMetadataRepository()

    event = {
        "scenario_id": "demo",
        "stage": "prod",
        "accepted": True,
        "reason": "admin token valid",
    }

    repo.save_admin_reload_event(event)
    event["accepted"] = False

    events = repo.list_admin_reload_events()
    assert len(events) == 1
    assert events[0]["accepted"] is True

    events[0]["accepted"] = False

    events_again = repo.list_admin_reload_events()
    assert events_again[0]["accepted"] is True


def test_metadata_db_url_reads_environment(monkeypatch):
    monkeypatch.delenv("IQA_METADATA_DB_URL", raising=False)
    assert metadata_db_url() is None

    monkeypatch.setenv("IQA_METADATA_DB_URL", "postgresql://iqa:secret@localhost:5432/iqa")
    assert metadata_db_url() == "postgresql://iqa:secret@localhost:5432/iqa"
