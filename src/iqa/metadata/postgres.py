"""PostgreSQL metadata repository foundation for IQA."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


SCHEMA_VERSION = "postgres_metadata_foundation_v001"

METADATA_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS metadata_schema_versions (
    schema_version TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS piece_events (
    piece_event_id TEXT PRIMARY KEY,
    source_event_id TEXT,
    scenario_id TEXT,
    lot_id TEXT,
    raw_dataset_id TEXT,
    manifest_id TEXT,
    dataset_version TEXT,
    replay_id TEXT,
    validation_id TEXT,
    scenario_version TEXT,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id TEXT PRIMARY KEY,
    piece_event_id TEXT NOT NULL,
    source_event_id TEXT,
    scenario_id TEXT NOT NULL,
    lot_id TEXT,
    raw_dataset_id TEXT,
    manifest_id TEXT,
    dataset_version TEXT,
    replay_id TEXT,
    validation_id TEXT,
    scenario_version TEXT,
    decision TEXT,
    model_version TEXT,
    roi_model_version TEXT,
    feedback_closed BOOLEAN NOT NULL DEFAULT false,
    feedback_closed_at TIMESTAMPTZ,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS feedback_events (
    prediction_id TEXT PRIMARY KEY,
    piece_event_id TEXT,
    scenario_id TEXT,
    feedback_source TEXT NOT NULL DEFAULT 'oracle_gt',
    eligible_for_train BOOLEAN,
    closed_at TIMESTAMPTZ,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS display_feedback_events (
    prediction_id TEXT PRIMARY KEY,
    piece_event_id TEXT,
    scenario_id TEXT,
    feedback_source TEXT NOT NULL DEFAULT 'human_sophie',
    eligible_for_train BOOLEAN,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS admin_reload_events (
    reload_event_id TEXT PRIMARY KEY,
    prediction_id TEXT,
    scenario_id TEXT NOT NULL,
    stage TEXT,
    reload_status TEXT,
    accepted BOOLEAN,
    registered_model_name TEXT,
    source_of_truth TEXT,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

METADATA_SCHEMA_VERSION_SQL = """
INSERT INTO metadata_schema_versions (schema_version)
VALUES (%s)
ON CONFLICT (schema_version) DO NOTHING;
"""


def _json_payload(record: dict[str, Any]) -> Jsonb:
    return Jsonb(deepcopy(record))


def _timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized)
        except ValueError:
            return None
    return None


def initialize_metadata_db(db_url: str) -> None:
    """Create or update the IQA metadata schema in PostgreSQL."""

    if not db_url:
        raise ValueError("IQA_METADATA_DB_URL is required to initialize PostgreSQL metadata.")

    with psycopg.connect(db_url) as connection:
        for statement in METADATA_SCHEMA_SQL.split(";"):
            sql = statement.strip()
            if sql:
                connection.execute(sql)
        connection.execute(METADATA_SCHEMA_VERSION_SQL, (SCHEMA_VERSION,))


class PostgresMetadataRepository:
    """PostgreSQL implementation of the IQA metadata repository protocol."""

    def __init__(self, db_url: str) -> None:
        if not db_url:
            raise ValueError("db_url is required for PostgresMetadataRepository.")
        self.db_url = db_url

    def save_piece_event(self, piece_event_id: str, record: dict[str, Any]) -> None:
        with psycopg.connect(self.db_url) as connection:
            connection.execute(
                """
                INSERT INTO piece_events (
                    piece_event_id, source_event_id, scenario_id, lot_id,
                    raw_dataset_id, manifest_id, dataset_version, replay_id,
                    validation_id, scenario_version, payload
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (piece_event_id) DO UPDATE SET
                    source_event_id = EXCLUDED.source_event_id,
                    scenario_id = EXCLUDED.scenario_id,
                    lot_id = EXCLUDED.lot_id,
                    raw_dataset_id = EXCLUDED.raw_dataset_id,
                    manifest_id = EXCLUDED.manifest_id,
                    dataset_version = EXCLUDED.dataset_version,
                    replay_id = EXCLUDED.replay_id,
                    validation_id = EXCLUDED.validation_id,
                    scenario_version = EXCLUDED.scenario_version,
                    payload = EXCLUDED.payload,
                    updated_at = now();
                """,
                (
                    piece_event_id,
                    record.get("source_event_id"),
                    record.get("scenario_id"),
                    record.get("lot_id"),
                    record.get("raw_dataset_id"),
                    record.get("manifest_id"),
                    record.get("dataset_version"),
                    record.get("replay_id"),
                    record.get("validation_id"),
                    record.get("scenario_version"),
                    _json_payload(record),
                ),
            )

    def get_piece_event(self, piece_event_id: str) -> dict[str, Any] | None:
        with psycopg.connect(self.db_url, row_factory=dict_row) as connection:
            row = connection.execute(
                "SELECT payload FROM piece_events WHERE piece_event_id = %s;",
                (piece_event_id,),
            ).fetchone()
        return deepcopy(row["payload"]) if row else None

    def save_prediction(self, prediction_id: str, record: dict[str, Any]) -> None:
        with psycopg.connect(self.db_url) as connection:
            connection.execute(
                """
                INSERT INTO predictions (
                    prediction_id, piece_event_id, source_event_id, scenario_id, lot_id,
                    raw_dataset_id, manifest_id, dataset_version, replay_id,
                    validation_id, scenario_version, decision, model_version,
                    roi_model_version, feedback_closed, feedback_closed_at, payload, created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, COALESCE(%s, now()))
                ON CONFLICT (prediction_id) DO UPDATE SET
                    piece_event_id = EXCLUDED.piece_event_id,
                    source_event_id = EXCLUDED.source_event_id,
                    scenario_id = EXCLUDED.scenario_id,
                    lot_id = EXCLUDED.lot_id,
                    raw_dataset_id = EXCLUDED.raw_dataset_id,
                    manifest_id = EXCLUDED.manifest_id,
                    dataset_version = EXCLUDED.dataset_version,
                    replay_id = EXCLUDED.replay_id,
                    validation_id = EXCLUDED.validation_id,
                    scenario_version = EXCLUDED.scenario_version,
                    decision = EXCLUDED.decision,
                    model_version = EXCLUDED.model_version,
                    roi_model_version = EXCLUDED.roi_model_version,
                    feedback_closed = EXCLUDED.feedback_closed,
                    feedback_closed_at = EXCLUDED.feedback_closed_at,
                    payload = EXCLUDED.payload,
                    updated_at = now();
                """,
                (
                    prediction_id,
                    record.get("piece_event_id"),
                    record.get("source_event_id"),
                    record.get("scenario_id"),
                    record.get("lot_id"),
                    record.get("raw_dataset_id"),
                    record.get("manifest_id"),
                    record.get("dataset_version"),
                    record.get("replay_id"),
                    record.get("validation_id"),
                    record.get("scenario_version"),
                    record.get("decision"),
                    record.get("model_version"),
                    record.get("roi_model_version"),
                    bool(record.get("feedback_closed", False)),
                    _timestamp(record.get("feedback_closed_at")),
                    _json_payload(record),
                    _timestamp(record.get("created_at")),
                ),
            )

    def get_prediction(self, prediction_id: str) -> dict[str, Any] | None:
        with psycopg.connect(self.db_url, row_factory=dict_row) as connection:
            row = connection.execute(
                "SELECT payload FROM predictions WHERE prediction_id = %s;",
                (prediction_id,),
            ).fetchone()
        return deepcopy(row["payload"]) if row else None

    def list_predictions(self) -> list[dict[str, Any]]:
        with psycopg.connect(self.db_url, row_factory=dict_row) as connection:
            rows = connection.execute(
                "SELECT payload FROM predictions ORDER BY created_at DESC, prediction_id DESC;"
            ).fetchall()
        return [deepcopy(row["payload"]) for row in rows]

    def save_feedback(self, prediction_id: str, record: dict[str, Any]) -> None:
        with psycopg.connect(self.db_url) as connection:
            connection.execute(
                """
                INSERT INTO feedback_events (
                    prediction_id, piece_event_id, scenario_id, feedback_source,
                    eligible_for_train, closed_at, payload
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (prediction_id) DO UPDATE SET
                    piece_event_id = EXCLUDED.piece_event_id,
                    scenario_id = EXCLUDED.scenario_id,
                    feedback_source = EXCLUDED.feedback_source,
                    eligible_for_train = EXCLUDED.eligible_for_train,
                    closed_at = EXCLUDED.closed_at,
                    payload = EXCLUDED.payload,
                    updated_at = now();
                """,
                (
                    prediction_id,
                    record.get("piece_event_id"),
                    record.get("scenario_id"),
                    record.get("feedback_source", "oracle_gt"),
                    record.get("eligible_for_train"),
                    _timestamp(record.get("closed_at")),
                    _json_payload(record),
                ),
            )

    def get_feedback(self, prediction_id: str) -> dict[str, Any] | None:
        with psycopg.connect(self.db_url, row_factory=dict_row) as connection:
            row = connection.execute(
                "SELECT payload FROM feedback_events WHERE prediction_id = %s;",
                (prediction_id,),
            ).fetchone()
        return deepcopy(row["payload"]) if row else None

    def save_display_feedback(self, prediction_id: str, record: dict[str, Any]) -> None:
        with psycopg.connect(self.db_url) as connection:
            connection.execute(
                """
                INSERT INTO display_feedback_events (
                    prediction_id, piece_event_id, scenario_id, feedback_source,
                    eligible_for_train, payload
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (prediction_id) DO UPDATE SET
                    piece_event_id = EXCLUDED.piece_event_id,
                    scenario_id = EXCLUDED.scenario_id,
                    feedback_source = EXCLUDED.feedback_source,
                    eligible_for_train = EXCLUDED.eligible_for_train,
                    payload = EXCLUDED.payload,
                    updated_at = now();
                """,
                (
                    prediction_id,
                    record.get("piece_event_id"),
                    record.get("scenario_id"),
                    record.get("feedback_source", "human_sophie"),
                    record.get("eligible_for_train"),
                    _json_payload(record),
                ),
            )

    def get_display_feedback(self, prediction_id: str) -> dict[str, Any] | None:
        with psycopg.connect(self.db_url, row_factory=dict_row) as connection:
            row = connection.execute(
                "SELECT payload FROM display_feedback_events WHERE prediction_id = %s;",
                (prediction_id,),
            ).fetchone()
        return deepcopy(row["payload"]) if row else None

    def mark_feedback_closed(self, prediction_id: str, closed_at: str) -> None:
        with psycopg.connect(self.db_url, row_factory=dict_row) as connection:
            row = connection.execute(
                "SELECT payload FROM predictions WHERE prediction_id = %s;",
                (prediction_id,),
            ).fetchone()
            if row is None:
                raise KeyError(prediction_id)
            payload = deepcopy(row["payload"])
            payload["feedback_closed"] = True
            payload["feedback_closed_at"] = closed_at
            connection.execute(
                """
                UPDATE predictions
                SET feedback_closed = true,
                    feedback_closed_at = %s,
                    payload = %s,
                    updated_at = now()
                WHERE prediction_id = %s;
                """,
                (_timestamp(closed_at), _json_payload(payload), prediction_id),
            )

    def save_admin_reload_event(self, record: dict[str, Any]) -> None:
        reload_event_id = record.get("reload_event_id")
        if not reload_event_id:
            raise ValueError("reload_event_id is required for admin reload events.")

        with psycopg.connect(self.db_url) as connection:
            connection.execute(
                """
                INSERT INTO admin_reload_events (
                    reload_event_id, prediction_id, scenario_id, stage,
                    reload_status, accepted, registered_model_name,
                    source_of_truth, payload, created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, COALESCE(%s, now()))
                ON CONFLICT (reload_event_id) DO NOTHING;
                """,
                (
                    reload_event_id,
                    record.get("prediction_id"),
                    record.get("scenario_id"),
                    record.get("stage"),
                    record.get("reload_status"),
                    record.get("accepted"),
                    record.get("registered_model_name"),
                    record.get("source_of_truth"),
                    _json_payload(record),
                    _timestamp(record.get("created_at")),
                ),
            )

    def list_admin_reload_events(self) -> list[dict[str, Any]]:
        with psycopg.connect(self.db_url, row_factory=dict_row) as connection:
            rows = connection.execute(
                "SELECT payload FROM admin_reload_events ORDER BY created_at ASC, reload_event_id ASC;"
            ).fetchall()
        return [deepcopy(row["payload"]) for row in rows]


__all__ = [
    "METADATA_SCHEMA_SQL",
    "SCHEMA_VERSION",
    "PostgresMetadataRepository",
    "initialize_metadata_db",
]
