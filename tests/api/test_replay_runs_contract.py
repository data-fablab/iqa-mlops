from __future__ import annotations

import pytest
from fastapi import HTTPException

from iqa.api.main import (
    REPLAY_RUN_STORE,
    app,
    create_replay_run,
    next_replay_event,
    reset_replay_run,
)
from iqa.api.schemas import ReplayRunRequest


@pytest.fixture(autouse=True)
def _clear_replay_runs() -> None:
    REPLAY_RUN_STORE.clear()
    yield
    REPLAY_RUN_STORE.clear()


def _event_ids(run_id: str, count: int) -> list[str]:
    ids: list[str] = []
    for _ in range(count):
        response = next_replay_event(run_id)
        assert response["event"] is not None
        ids.append(response["event"]["piece_event_id"])
    return ids


def test_replay_run_routes_are_registered() -> None:
    route_paths = {route.path for route in app.routes}

    assert "/replay-runs" in route_paths
    assert "/replay-runs/{replay_run_id}/next" in route_paths
    assert "/replay-runs/{replay_run_id}/reset" in route_paths


def test_replay_runs_with_same_inputs_produce_same_order() -> None:
    first = create_replay_run(ReplayRunRequest(scenario_id="production_replay_natural"))
    second = create_replay_run(ReplayRunRequest(scenario_id="production_replay_natural"))

    assert first["replay_run_id"] != second["replay_run_id"]
    assert first["total_events"] == second["total_events"]
    assert _event_ids(first["replay_run_id"], 5) == _event_ids(second["replay_run_id"], 5)


def test_replay_run_reset_is_isolated_by_run() -> None:
    first = create_replay_run(ReplayRunRequest(scenario_id="production_replay_natural"))
    second = create_replay_run(ReplayRunRequest(scenario_id="production_replay_natural"))

    first_initial = next_replay_event(first["replay_run_id"])["event"]["piece_event_id"]
    second_initial = next_replay_event(second["replay_run_id"])["event"]["piece_event_id"]
    second_next = next_replay_event(second["replay_run_id"])["event"]["piece_event_id"]

    reset = reset_replay_run(first["replay_run_id"])
    first_after_reset = next_replay_event(first["replay_run_id"])["event"]["piece_event_id"]
    second_after_first_reset = next_replay_event(second["replay_run_id"])["event"]["piece_event_id"]

    assert reset["cursor"] == 0
    assert first_after_reset == first_initial
    assert second_initial == first_initial
    assert second_after_first_reset != second_initial
    assert second_after_first_reset != second_next


def test_drift_replay_run_is_supported() -> None:
    run = create_replay_run(ReplayRunRequest(scenario_id="drift_domain_extension"))
    response = next_replay_event(run["replay_run_id"])

    assert response["event"]["scenario_id"] == "drift_domain_extension"
    assert response["event"]["dataset_version"] == "drift_domain_extension_v001"


def test_replay_run_unknown_ids_return_structured_404() -> None:
    with pytest.raises(HTTPException) as scenario_error:
        create_replay_run(ReplayRunRequest(scenario_id="unknown_scenario"))

    with pytest.raises(HTTPException) as run_error:
        next_replay_event("unknown_run")

    assert scenario_error.value.status_code == 404
    assert scenario_error.value.detail["error_code"] == "replay_scenario_not_found"
    assert run_error.value.status_code == 404
    assert run_error.value.detail["error_code"] == "replay_run_not_found"
