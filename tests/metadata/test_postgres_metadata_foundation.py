from __future__ import annotations

import os
from uuid import uuid4

import pytest

from iqa.metadata.postgres import METADATA_SCHEMA_SQL, PostgresMetadataRepository, initialize_metadata_db
from iqa.metadata.repository import MemoryMetadataRepository, create_metadata_repository


def test_metadata_repository_factory_defaults_to_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IQA_METADATA_BACKEND", raising=False)
    monkeypatch.delenv("IQA_METADATA_DB_URL", raising=False)

    assert isinstance(create_metadata_repository(), MemoryMetadataRepository)


def test_metadata_repository_factory_requires_url_for_postgres(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IQA_METADATA_BACKEND", "postgres")
    monkeypatch.delenv("IQA_METADATA_DB_URL", raising=False)

    with pytest.raises(RuntimeError, match="IQA_METADATA_DB_URL is required"):
        create_metadata_repository()


def test_metadata_repository_factory_rejects_unknown_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IQA_METADATA_BACKEND", "sqlite")

    with pytest.raises(RuntimeError, match="Unsupported IQA_METADATA_BACKEND"):
        create_metadata_repository()


def test_postgres_schema_declares_expected_tables_without_sqlite() -> None:
    schema = METADATA_SCHEMA_SQL.lower()

    for table in [
        "metadata_schema_versions",
        "piece_events",
        "predictions",
        "feedback_events",
        "display_feedback_events",
        "admin_reload_events",
        "lot_events",
        "incident_events",
        "model_version_events",
        "scenario_version_events",
        "lifecycle_trigger_events",
    ]:
        assert f"create table if not exists {table}" in schema
    assert "jsonb" in schema
    assert "sqlite" not in schema


def test_memory_repository_keeps_extended_metadata_events() -> None:
    repo = MemoryMetadataRepository()

    repo.save_lot_event({"lot_id": "lot_001", "scenario_id": "production_replay_natural"})
    repo.save_incident_event({"incident_id": "incident_001", "incident_type": "false_negative"})
    repo.save_model_version_event(
        {
            "model_version_event_id": "model_event_001",
            "model_version": "feature_ae_v002",
            "dataset_version": "feature_ae_good_v002",
        }
    )
    repo.save_scenario_version_event(
        {
            "scenario_version_id": "scenario_event_001",
            "scenario_id": "production_replay_natural",
            "scenario_version": "production_replay_natural_v001",
        }
    )
    repo.save_lifecycle_trigger_event(
        {
            "lifecycle_trigger_event_id": "trigger_001",
            "scenario_id": "production_replay_natural",
            "trigger_reason": "natural_50_oracle_conformes",
            "dataset_version": "feature_ae_good_v002",
            "manifest_version": "feature_ae_good_v002_manifest_v001",
        }
    )

    assert repo.list_lot_events()[0]["lot_id"] == "lot_001"
    assert repo.list_incident_events()[0]["incident_id"] == "incident_001"
    assert repo.list_model_version_events()[0]["dataset_version"] == "feature_ae_good_v002"
    assert repo.list_scenario_version_events()[0]["scenario_version"] == "production_replay_natural_v001"
    assert repo.list_lifecycle_trigger_events()[0]["trigger_reason"] == "natural_50_oracle_conformes"


@pytest.fixture(scope="module")
def postgres_repo() -> PostgresMetadataRepository:
    db_url = os.getenv("IQA_METADATA_DB_URL")
    if not db_url:
        pytest.skip("IQA_METADATA_DB_URL is not set.")

    initialize_metadata_db(db_url)
    return PostgresMetadataRepository(db_url)


@pytest.mark.postgres_contract
def test_postgres_repository_saves_lists_and_updates_predictions(postgres_repo: PostgresMetadataRepository) -> None:
    suffix = uuid4().hex
    prediction_id = f"pred_{suffix}"
    record = {
        "prediction_id": prediction_id,
        "piece_event_id": f"sim_event_{suffix}",
        "source_event_id": f"piece_event_{suffix}",
        "scenario_id": "production_replay_natural",
        "lot_id": "lot_pg_001",
        "raw_dataset_id": "hss_iad_casting_raw_v1",
        "manifest_id": "casting_flux_replay_plan_natural_v001",
        "dataset_version": "production_replay_natural_v001",
        "replay_id": "production_replay_natural_v001",
        "validation_id": None,
        "scenario_version": "production_replay_natural_v001",
        "decision": "Vert",
        "model_version": "rd_feature_ae_gated_v001_bootstrap",
        "roi_model_version": "roi_segmenter_v001_fixed",
        "feedback_closed": False,
        "created_at": "2026-06-16T12:00:00+00:00",
    }

    postgres_repo.save_prediction(prediction_id, record)
    record["decision"] = "Orange"
    postgres_repo.save_prediction(prediction_id, record)

    saved = postgres_repo.get_prediction(prediction_id)
    assert saved is not None
    assert saved["decision"] == "Orange"
    assert saved["piece_event_id"] == f"sim_event_{suffix}"
    assert saved["source_event_id"] == f"piece_event_{suffix}"
    assert any(row["prediction_id"] == prediction_id for row in postgres_repo.list_predictions())


@pytest.mark.postgres_contract
def test_postgres_repository_keeps_oracle_and_display_feedback_separate(
    postgres_repo: PostgresMetadataRepository,
) -> None:
    suffix = uuid4().hex
    prediction_id = f"pred_{suffix}"
    postgres_repo.save_prediction(
        prediction_id,
        {
            "prediction_id": prediction_id,
            "piece_event_id": f"piece_{suffix}",
            "scenario_id": "demo",
            "feedback_closed": False,
        },
    )

    postgres_repo.save_display_feedback(
        prediction_id,
        {
            "prediction_id": prediction_id,
            "piece_event_id": f"piece_{suffix}",
            "scenario_id": "demo",
            "feedback_source": "human_sophie",
            "eligible_for_train": False,
        },
    )
    postgres_repo.save_feedback_and_close_prediction(
        prediction_id,
        {
            "prediction_id": prediction_id,
            "piece_event_id": f"piece_{suffix}",
            "scenario_id": "demo",
            "feedback_source": "oracle_gt",
            "eligible_for_train": True,
            "closed_at": "2026-06-16T12:05:00+00:00",
        },
        "2026-06-16T12:05:00+00:00",
    )

    assert postgres_repo.get_display_feedback(prediction_id)["feedback_source"] == "human_sophie"
    assert postgres_repo.get_feedback(prediction_id)["feedback_source"] == "oracle_gt"
    assert postgres_repo.get_prediction(prediction_id)["feedback_closed"] is True


@pytest.mark.postgres_contract
def test_postgres_repository_appends_admin_reload_events(postgres_repo: PostgresMetadataRepository) -> None:
    suffix = uuid4().hex
    first = {
        "reload_event_id": f"reload_{suffix}_1",
        "scenario_id": "demo",
        "stage": "prod",
        "reload_status": "accepted",
        "accepted": True,
        "reason": "admin token valid",
        "registered_model_name": "feature_ae__demo",
        "source_of_truth": "mlflow_registry",
        "created_at": "2026-06-16T12:10:00+00:00",
    }
    second = {**first, "reload_event_id": f"reload_{suffix}_2", "accepted": False}

    postgres_repo.save_admin_reload_event(first)
    postgres_repo.save_admin_reload_event(second)

    event_ids = {event["reload_event_id"] for event in postgres_repo.list_admin_reload_events()}
    assert first["reload_event_id"] in event_ids
    assert second["reload_event_id"] in event_ids


@pytest.mark.postgres_contract
def test_postgres_repository_keeps_replay_piece_and_source_identity_distinct(
    postgres_repo: PostgresMetadataRepository,
) -> None:
    suffix = uuid4().hex
    postgres_repo.save_piece_event(
        f"sim_event_{suffix}",
        {
            "piece_event_id": f"sim_event_{suffix}",
            "source_event_id": f"piece_event_{suffix}",
            "scenario_id": "production_replay_natural",
            "raw_dataset_id": "hss_iad_casting_raw_v1",
            "manifest_id": "casting_flux_replay_plan_natural_v001",
            "dataset_version": "production_replay_natural_v001",
            "replay_id": "production_replay_natural_v001",
            "scenario_version": "production_replay_natural_v001",
        },
    )
    piece_event = postgres_repo.get_piece_event(f"sim_event_{suffix}")
    assert piece_event["piece_event_id"] == f"sim_event_{suffix}"
    assert piece_event["source_event_id"] == f"piece_event_{suffix}"

    postgres_repo.save_prediction(
        f"pred_{suffix}",
        {
            "prediction_id": f"pred_{suffix}",
            "piece_event_id": f"sim_event_{suffix}",
            "source_event_id": f"piece_event_{suffix}",
            "scenario_id": "production_replay_natural",
            "feedback_closed": False,
        },
    )

    saved = postgres_repo.get_prediction(f"pred_{suffix}")
    assert saved["piece_event_id"] == f"sim_event_{suffix}"
    assert saved["source_event_id"] == f"piece_event_{suffix}"
    assert saved["piece_event_id"] != saved["source_event_id"]


@pytest.mark.postgres_contract
def test_postgres_repository_persists_extended_metadata_events(
    postgres_repo: PostgresMetadataRepository,
) -> None:
    suffix = uuid4().hex
    trigger_event_id = f"trigger_{suffix}"

    postgres_repo.save_lot_event(
        {
            "lot_id": f"lot_{suffix}",
            "scenario_id": "production_replay_natural",
            "dataset_version": "production_replay_natural_v001",
            "source_class": "Casting_class1",
            "status": "served",
        }
    )
    postgres_repo.save_incident_event(
        {
            "incident_id": f"incident_{suffix}",
            "incident_type": "false_negative",
            "severity": "high",
            "scenario_id": "production_replay_natural",
            "lot_id": f"lot_{suffix}",
            "prediction_id": f"pred_{suffix}",
            "dataset_version": "production_replay_natural_v001",
        }
    )
    postgres_repo.save_model_version_event(
        {
            "model_version_event_id": f"model_event_{suffix}",
            "registered_model_name": "feature_ae__production_replay_natural",
            "model_version": "feature_ae_v002",
            "scenario_id": "production_replay_natural",
            "stage": "candidate",
            "source_of_truth": "mlflow_registry",
            "artifact_uri": "s3://mlflow-artifacts/model",
            "dataset_version": "feature_ae_good_v002",
            "manifest_version": "feature_ae_good_v002_manifest_v001",
        }
    )
    postgres_repo.save_scenario_version_event(
        {
            "scenario_version_id": f"scenario_event_{suffix}",
            "scenario_id": "production_replay_natural",
            "scenario_version": "production_replay_natural_v001",
            "dataset_version": "production_replay_natural_v001",
            "replay_id": "production_replay_natural_v001",
            "lifecycle_status": "candidate_ready",
        }
    )
    postgres_repo.save_lifecycle_trigger_event(
        {
            "lifecycle_trigger_event_id": trigger_event_id,
            "scenario_id": "production_replay_natural",
            "trigger_reason": "natural_50_oracle_conformes",
            "trigger_lifecycle": True,
            "dataset_version": "feature_ae_good_v002",
            "manifest_version": "feature_ae_good_v002_manifest_v001",
            "model_version": "feature_ae_v002",
            "lot_id": f"lot_{suffix}",
        }
    )

    trigger_events = {
        event["lifecycle_trigger_event_id"]: event
        for event in postgres_repo.list_lifecycle_trigger_events()
    }
    assert trigger_events[trigger_event_id]["scenario_id"] == "production_replay_natural"
    assert trigger_events[trigger_event_id]["trigger_reason"] == "natural_50_oracle_conformes"
    assert trigger_events[trigger_event_id]["dataset_version"] == "feature_ae_good_v002"
    assert trigger_events[trigger_event_id]["manifest_version"] == "feature_ae_good_v002_manifest_v001"
    assert any(event["lot_id"] == f"lot_{suffix}" for event in postgres_repo.list_lot_events())
    assert any(event["incident_id"] == f"incident_{suffix}" for event in postgres_repo.list_incident_events())
    assert any(
        event["model_version_event_id"] == f"model_event_{suffix}"
        for event in postgres_repo.list_model_version_events()
    )
    assert any(
        event["scenario_version_id"] == f"scenario_event_{suffix}"
        for event in postgres_repo.list_scenario_version_events()
    )
