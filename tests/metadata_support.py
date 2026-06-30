"""Helpers to observe and mutate API metadata through the repository seam.

The API no longer keeps parallel in-memory globals (``PREDICTION_STORE`` and
friends): the single ``MetadataRepository`` owned by the API is the only store.
These helpers let tests read and adjust that store without reaching for the old
globals, so a "memory" run and a "postgres" run observe the same behavior.
"""

from __future__ import annotations

from typing import Any

from iqa.api import main as api


def repository() -> Any:
    """Return the single metadata repository owned by the API."""

    return api.metadata_repository()


def get_prediction(prediction_id: str) -> dict[str, Any] | None:
    return repository().get_prediction(prediction_id)


def get_feedback(prediction_id: str) -> dict[str, Any] | None:
    return repository().get_feedback(prediction_id)


def get_display_feedback(prediction_id: str) -> dict[str, Any] | None:
    return repository().get_display_feedback(prediction_id)


def list_incident_events() -> list[dict[str, Any]]:
    return repository().list_incident_events()


def list_admin_reload_events() -> list[dict[str, Any]]:
    return repository().list_admin_reload_events()


def set_prediction_field(prediction_id: str, field: str, value: Any) -> None:
    """Mutate a stored prediction record through the repository.

    The memory adapter returns copies, so a field change must be written back to
    take effect (mirrors how a persisted store behaves).
    """

    record = repository().get_prediction(prediction_id)
    if record is None:
        raise KeyError(prediction_id)
    record[field] = value
    repository().save_prediction(prediction_id, record)
