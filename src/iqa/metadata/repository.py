"""Repository foundation for IQA metadata records.

This module is intentionally dependency free for NAT01.
It prepares the metadata boundary before a PostgreSQL backend is added.
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Protocol


class MetadataRepository(Protocol):
    """Protocol for IQA metadata persistence."""

    def save_prediction(self, prediction_id: str, record: dict[str, Any]) -> None:
        """Store a prediction metadata record."""

    def get_prediction(self, prediction_id: str) -> dict[str, Any] | None:
        """Return a prediction metadata record."""

    def list_predictions(self) -> list[dict[str, Any]]:
        """Return prediction metadata records."""

    def save_feedback(self, prediction_id: str, record: dict[str, Any]) -> None:
        """Store an oracle feedback metadata record."""

    def get_feedback(self, prediction_id: str) -> dict[str, Any] | None:
        """Return an oracle feedback metadata record."""

    def save_display_feedback(self, prediction_id: str, record: dict[str, Any]) -> None:
        """Store a display only human feedback metadata record."""

    def get_display_feedback(self, prediction_id: str) -> dict[str, Any] | None:
        """Return a display only human feedback metadata record."""

    def mark_feedback_closed(self, prediction_id: str, closed_at: str) -> None:
        """Mark a prediction as closed for oracle feedback."""

    def save_admin_reload_event(self, record: dict[str, Any]) -> None:
        """Store an admin reload audit event."""

    def list_admin_reload_events(self) -> list[dict[str, Any]]:
        """Return admin reload audit events."""


class MemoryMetadataRepository:
    """In memory implementation used by the API and tests before PostgreSQL."""

    def __init__(self) -> None:
        self._predictions: dict[str, dict[str, Any]] = {}
        self._feedbacks: dict[str, dict[str, Any]] = {}
        self._display_feedbacks: dict[str, dict[str, Any]] = {}
        self._admin_reload_events: list[dict[str, Any]] = []

    def save_prediction(self, prediction_id: str, record: dict[str, Any]) -> None:
        self._predictions[prediction_id] = deepcopy(record)

    def get_prediction(self, prediction_id: str) -> dict[str, Any] | None:
        record = self._predictions.get(prediction_id)
        return deepcopy(record) if record is not None else None

    def list_predictions(self) -> list[dict[str, Any]]:
        return [deepcopy(record) for record in self._predictions.values()]

    def save_feedback(self, prediction_id: str, record: dict[str, Any]) -> None:
        self._feedbacks[prediction_id] = deepcopy(record)

    def get_feedback(self, prediction_id: str) -> dict[str, Any] | None:
        record = self._feedbacks.get(prediction_id)
        return deepcopy(record) if record is not None else None

    def save_display_feedback(self, prediction_id: str, record: dict[str, Any]) -> None:
        self._display_feedbacks[prediction_id] = deepcopy(record)

    def get_display_feedback(self, prediction_id: str) -> dict[str, Any] | None:
        record = self._display_feedbacks.get(prediction_id)
        return deepcopy(record) if record is not None else None

    def mark_feedback_closed(self, prediction_id: str, closed_at: str) -> None:
        record = self._predictions[prediction_id]
        record["feedback_closed"] = True
        record["feedback_closed_at"] = closed_at

    def save_admin_reload_event(self, record: dict[str, Any]) -> None:
        self._admin_reload_events.append(deepcopy(record))

    def list_admin_reload_events(self) -> list[dict[str, Any]]:
        return [deepcopy(record) for record in self._admin_reload_events]


def metadata_db_url() -> str | None:
    """Return the optional IQA metadata database URL."""

    return os.getenv("IQA_METADATA_DB_URL")
