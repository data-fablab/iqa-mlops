"""Tests for the iqa-run-rollback boundary (Issue 5).

The CLI orchestrates the existing rollback path; the restoration logic itself
(rollback.py) is covered by its own tests. Here we check the orchestration:
faulty version resolution and delegation to rollback_model.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from scripts.run_rollback import run_rollback

pytestmark = pytest.mark.unit


def test_resolves_current_prod_as_faulty_then_rolls_back() -> None:
    with patch("scripts.run_rollback.resolve_model_artifacts", return_value={"version": "3"}) as mock_resolve, \
         patch("scripts.run_rollback.rollback_model", return_value={"success": True, "previous_prod_version": "2", "faulty_version_archived": "3"}) as mock_rollback:
        payload = run_rollback("drift_domain_extension", tracking_uri="http://mlflow:5000")

    mock_resolve.assert_called_once()
    assert mock_rollback.call_args.kwargs["faulty_version"] == "3"
    assert payload["registered_model_name"] == "feature_ae__drift_domain_extension"
    assert payload["faulty_version"] == "3"
    assert payload["status"] == "rolled_back"
    assert payload["success"] is True


def test_explicit_faulty_version_skips_prod_resolution() -> None:
    with patch("scripts.run_rollback.resolve_model_artifacts") as mock_resolve, \
         patch("scripts.run_rollback.rollback_model", return_value={"success": True}) as mock_rollback:
        run_rollback("drift_domain_extension", faulty_version="5")

    mock_resolve.assert_not_called()
    assert mock_rollback.call_args.kwargs["faulty_version"] == "5"


def test_failed_rollback_is_reported_as_failed() -> None:
    with patch("scripts.run_rollback.resolve_model_artifacts", return_value={"version": "3"}), \
         patch("scripts.run_rollback.rollback_model", return_value={"success": False, "error": "boom"}):
        payload = run_rollback("drift_domain_extension")

    assert payload["status"] == "failed"
    assert payload["success"] is False
