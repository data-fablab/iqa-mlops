from __future__ import annotations

from collections.abc import Iterator
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from iqa.metadata.postgres import (
    PostgresMetadataRepository,
    initialize_metadata_db,
)
from iqa.monitoring.lifecycle_signals import (
    collect_and_record_lifecycle_signal,
)


@pytest.fixture
def nat16_postgres_url(
    isolated_postgres_db_url: str,
) -> Iterator[str]:
    params = conninfo_to_dict(isolated_postgres_db_url)
    params.pop("options", None)

    admin_url = make_conninfo(**params)
    schema = f"iqa_nat16_{uuid4().hex}"

    with psycopg.connect(admin_url) as connection:
        connection.execute(
            sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema))
        )

    params["options"] = f"-c search_path={schema}"
    db_url = make_conninfo(**params)
    initialize_metadata_db(db_url)

    try:
        yield db_url
    finally:
        with psycopg.connect(admin_url) as connection:
            connection.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    sql.Identifier(schema)
                )
            )


def _save_natural_prediction(
    repository: PostgresMetadataRepository,
    index: int,
    *,
    roi_status: str = "ok",
) -> str:
    prediction_id = f"nat16_prediction_{index:03d}"
    closed_at = f"2026-06-27T10:{index % 60:02d}:00+00:00"

    repository.save_prediction(
        prediction_id,
        {
            "prediction_id": prediction_id,
            "piece_event_id": f"nat16_piece_{index:03d}",
            "scenario_id": "production_replay_natural",
            "lot_id": "nat16_natural_lot_001",
            "decision": "Vert",
            "model_version": "feature_ae_v001",
            "roi_model_version": "roi_segmenter_v001",
            "roi_status": roi_status,
            "feedback_closed": False,
            "created_at": f"2026-06-27T09:{index % 60:02d}:00+00:00",
        },
    )
    repository.save_feedback_and_close_prediction(
        prediction_id,
        {
            "prediction_id": prediction_id,
            "piece_event_id": f"nat16_piece_{index:03d}",
            "scenario_id": "production_replay_natural",
            "feedback_source": "oracle_gt",
            "feedback_closed": True,
            "eligible_for_train": True,
            "verdict": {"verdict": "conforme"},
            "closed_at": closed_at,
        },
        closed_at,
    )
    return prediction_id


@pytest.mark.postgres_contract
def test_postgres_natural_signal_triggers_at_50_and_survives_restart(
    nat16_postgres_url: str,
) -> None:
    repository = PostgresMetadataRepository(nat16_postgres_url)

    for index in range(49):
        _save_natural_prediction(
            repository,
            index,
            roi_status="fail" if index < 5 else "ok",
        )

    waiting = collect_and_record_lifecycle_signal(
        repository,
        scenario_id="production_replay_natural",
        roi_window_size=49,
    )

    assert waiting["trigger_lifecycle"] is False
    assert waiting["signal"]["conforming_validated_count"] == 49
    assert waiting["signal"]["roi_fail_rate"] == pytest.approx(5 / 49)

    _save_natural_prediction(repository, 49)

    restarted_repository = PostgresMetadataRepository(nat16_postgres_url)
    triggered = collect_and_record_lifecycle_signal(
        restarted_repository,
        scenario_id="production_replay_natural",
        roi_window_size=50,
    )

    assert triggered["trigger_lifecycle"] is True
    assert triggered["signal"]["conforming_validated_count"] == 50
    assert triggered["lifecycle_decision"]["trigger_reason"] == (
        "natural_50_oracle_conformes"
    )
    assert len(triggered["consumed_prediction_ids"]) == 50

    persisted_events = (
        restarted_repository.list_lifecycle_trigger_events()
    )
    persisted_trigger = next(
        event
        for event in persisted_events
        if event["lifecycle_trigger_event_id"]
        == triggered["lifecycle_trigger_event_id"]
    )
    assert len(persisted_trigger["consumed_prediction_ids"]) == 50
    assert persisted_trigger["watermark"]["latest_prediction_id"] is not None

    second_restart = PostgresMetadataRepository(nat16_postgres_url)
    replayed = collect_and_record_lifecycle_signal(
        second_restart,
        scenario_id="production_replay_natural",
        roi_window_size=50,
    )

    assert replayed["trigger_lifecycle"] is False
    assert replayed["signal"]["conforming_validated_count"] == 0
    assert replayed["consumed_prediction_ids"] == []


@pytest.mark.postgres_contract
def test_postgres_versioned_drift_triggers_once_after_confirmation(
    nat16_postgres_url: str,
) -> None:
    repository = PostgresMetadataRepository(nat16_postgres_url)
    drift_event_id = "nat16_drift_observation_001"

    repository.save_scenario_version_event(
        {
            "scenario_version_id": drift_event_id,
            "scenario_id": "drift_domain_extension",
            "scenario_version": "drift_domain_extension_v001",
            "dataset_version": "drift_domain_extension_v001",
            "manifest_version": "drift_manifest_v001",
            "replay_id": "nat16_drift_replay_001",
            "lifecycle_status": "monitoring_green",
            "drift_confirmed": False,
        }
    )

    waiting = collect_and_record_lifecycle_signal(
        repository,
        scenario_id="drift_domain_extension",
    )

    assert waiting["trigger_lifecycle"] is False
    assert waiting["signal"]["drift_confirmed"] is False

    repository.save_scenario_version_event(
        {
            "scenario_version_id": drift_event_id,
            "scenario_id": "drift_domain_extension",
            "scenario_version": "drift_domain_extension_v001",
            "dataset_version": "drift_domain_extension_v001",
            "manifest_version": "drift_manifest_v001",
            "replay_id": "nat16_drift_replay_001",
            "lifecycle_status": "drift_confirmed",
            "drift_confirmed": True,
        }
    )

    restarted_repository = PostgresMetadataRepository(nat16_postgres_url)
    triggered = collect_and_record_lifecycle_signal(
        restarted_repository,
        scenario_id="drift_domain_extension",
    )

    assert triggered["trigger_lifecycle"] is True
    assert triggered["signal"]["drift_confirmed"] is True
    assert triggered["consumed_drift_event_ids"] == [drift_event_id]
    assert triggered["lifecycle_decision"]["trigger_reason"] == (
        "drift_confirmed"
    )

    second_restart = PostgresMetadataRepository(nat16_postgres_url)
    replayed = collect_and_record_lifecycle_signal(
        second_restart,
        scenario_id="drift_domain_extension",
    )

    assert replayed["trigger_lifecycle"] is False
    assert replayed["signal"]["drift_confirmed"] is False
    assert replayed["watermark"]["drift_event_consumed"] is True
