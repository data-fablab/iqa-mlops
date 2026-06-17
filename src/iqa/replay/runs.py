"""File-backed replay runs for Phase 2 API scheduling."""

from __future__ import annotations

import csv
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from iqa.metadata.contracts import MANIFEST_CONTRACTS


BASE_DIR = Path(__file__).resolve().parents[3]
REPLAY_CONTRACTS_BY_SCENARIO = {
    contract.scenario_id: contract
    for contract in MANIFEST_CONTRACTS.values()
    if contract.kind == "replay" and contract.scenario_id is not None
}


def _int_value(value: str | None) -> int:
    try:
        return int(value or "0")
    except ValueError:
        return 0


class FileBackedReplayRepository:
    """Read replay events from deterministic CSV manifests."""

    def __init__(self, *, base_dir: Path = BASE_DIR) -> None:
        self.base_dir = base_dir

    def list_events(self, scenario_id: str) -> list[dict[str, Any]]:
        contract = REPLAY_CONTRACTS_BY_SCENARIO.get(scenario_id)
        if contract is None:
            raise KeyError(scenario_id)

        path = self.base_dir / contract.path
        with path.open(newline="", encoding="utf-8") as file:
            events = [dict(row) for row in csv.DictReader(file) if row.get("scenario_id") == scenario_id]

        events.sort(
            key=lambda row: (
                _int_value(row.get("sequence_number")),
                row.get("scheduled_at") or "",
                row.get("piece_event_id") or "",
            )
        )
        return events


@dataclass
class ReplayRunState:
    replay_run_id: str
    scenario_id: str
    events: list[dict[str, Any]]
    cursor: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def total_events(self) -> int:
        return len(self.events)


class ReplayRunStore:
    """In-memory replay run scheduler state.

    The backing repository is file-based today and can be replaced by PostgreSQL
    later without changing the API route semantics.
    """

    def __init__(self, repository: FileBackedReplayRepository | None = None) -> None:
        self.repository = repository or FileBackedReplayRepository()
        self._runs: dict[str, ReplayRunState] = {}

    def clear(self) -> None:
        self._runs.clear()

    def create_run(self, scenario_id: str) -> dict[str, Any]:
        events = self.repository.list_events(scenario_id)
        replay_run_id = f"replay_run_{uuid4().hex}"
        state = ReplayRunState(
            replay_run_id=replay_run_id,
            scenario_id=scenario_id,
            events=events,
        )
        self._runs[replay_run_id] = state
        return self._run_response(state)

    def next_event(self, replay_run_id: str) -> dict[str, Any]:
        state = self._get_run(replay_run_id)
        if state.cursor >= state.total_events:
            return self._next_response(state, event=None)

        event = deepcopy(state.events[state.cursor])
        state.cursor += 1
        return self._next_response(state, event=event)

    def reset_run(self, replay_run_id: str) -> dict[str, Any]:
        state = self._get_run(replay_run_id)
        state.cursor = 0
        return self._run_response(state)

    def _get_run(self, replay_run_id: str) -> ReplayRunState:
        try:
            return self._runs[replay_run_id]
        except KeyError as exc:
            raise KeyError(replay_run_id) from exc

    def _run_response(self, state: ReplayRunState) -> dict[str, Any]:
        return {
            "replay_run_id": state.replay_run_id,
            "scenario_id": state.scenario_id,
            "cursor": state.cursor,
            "total_events": state.total_events,
            "created_at": state.created_at,
            "finished": state.cursor >= state.total_events,
        }

    def _next_response(self, state: ReplayRunState, *, event: dict[str, Any] | None) -> dict[str, Any]:
        return {
            **self._run_response(state),
            "event": event,
            "finished": event is None and state.cursor >= state.total_events,
        }


__all__ = [
    "FileBackedReplayRepository",
    "REPLAY_CONTRACTS_BY_SCENARIO",
    "ReplayRunStore",
]
