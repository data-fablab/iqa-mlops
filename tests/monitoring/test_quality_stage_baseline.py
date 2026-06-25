"""Tests for record_prod_promotion_quality (Issue 5).

At a prod promotion the quality baseline must advance so the regression rule can
compare prod vs previous_prod: exactly one previous_prod, the new model logged as
prod. MLflow is faked (no live server).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from iqa.monitoring import model_metrics as mm
from iqa.monitoring.model_metrics import TAG_MODEL_VERSION, TAG_STAGE

pytestmark = pytest.mark.unit


def _run(run_id: str, *, model_version: str, stage: str) -> SimpleNamespace:
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id),
        data=SimpleNamespace(tags={TAG_MODEL_VERSION: model_version, TAG_STAGE: stage}),
    )


class FakeClient:
    """Minimal MlflowClient capturing set_tag and answering search_runs by stage."""

    def __init__(self, runs_by_stage: dict[str, list[SimpleNamespace]]):
        self._runs_by_stage = runs_by_stage
        self.set_tags: list[tuple[str, str, str]] = []

    def get_experiment_by_name(self, name: str):
        return SimpleNamespace(experiment_id="1")

    def search_runs(self, _ids, filter_string="", order_by=None, max_results=1000):
        for stage, runs in self._runs_by_stage.items():
            if f"'{stage}'" in filter_string:
                return list(runs)
        return []

    def set_tag(self, run_id: str, key: str, value: str) -> None:
        self.set_tags.append((run_id, key, value))


def _record(client: FakeClient, **kwargs):
    with patch("mlflow.tracking.MlflowClient", return_value=client), \
         patch("iqa.monitoring.model_metrics.log_model_quality_metrics", return_value="new_prod_run") as mock_log:
        result = mm.record_prod_promotion_quality(
            {"image_ap": 0.9}, model_version="new_cand", **kwargs
        )
    return result, mock_log


def test_demotes_current_prod_to_previous_prod_and_logs_new_prod() -> None:
    client = FakeClient({
        "previous_prod": [],
        "prod": [_run("r_old_prod", model_version="old_cand", stage="prod")],
    })
    result, mock_log = _record(client)

    # Old prod demoted to previous_prod.
    assert ("r_old_prod", TAG_STAGE, "previous_prod") in client.set_tags
    # New prod logged with the promoted model version.
    assert mock_log.call_args.kwargs["stage"] == "prod"
    assert mock_log.call_args.kwargs["model_version"] == "new_cand"
    assert result["previous_prod_model_version"] == "old_cand"
    assert result["prod_run_id"] == "new_prod_run"


def test_archives_stale_previous_prod_to_keep_single_baseline() -> None:
    client = FakeClient({
        "previous_prod": [_run("r_stale", model_version="ancient", stage="previous_prod")],
        "prod": [_run("r_prod", model_version="old_cand", stage="prod")],
    })
    _record(client)

    # The stale previous_prod is archived (so only one previous_prod remains).
    assert ("r_stale", TAG_STAGE, "archived") in client.set_tags
    assert ("r_prod", TAG_STAGE, "previous_prod") in client.set_tags


def test_extra_prod_runs_are_archived_not_kept_as_previous_prod() -> None:
    client = FakeClient({
        "previous_prod": [],
        "prod": [
            _run("r_latest", model_version="latest", stage="prod"),
            _run("r_extra", model_version="extra", stage="prod"),
        ],
    })
    result, _ = _record(client)

    assert ("r_latest", TAG_STAGE, "previous_prod") in client.set_tags
    assert ("r_extra", TAG_STAGE, "archived") in client.set_tags
    assert result["previous_prod_model_version"] == "latest"


def test_first_ever_promotion_has_no_previous_prod() -> None:
    client = FakeClient({"previous_prod": [], "prod": []})
    result, mock_log = _record(client)

    assert result["previous_prod_model_version"] is None
    assert mock_log.call_args.kwargs["stage"] == "prod"
