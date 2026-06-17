from __future__ import annotations

from pathlib import Path


def test_replay_runbook_documents_api_and_airflow_checks() -> None:
    content = Path("docs/replay-runbook.md").read_text(encoding="utf-8")

    for term in [
        "/replay-scenarios",
        "/replay-runs",
        "/replay-runs/<replay_run_id>/next",
        "/replay-runs/<replay_run_id>/reset",
        "production_replay_natural",
        "drift_domain_extension",
        "airflow dags list",
        "airflow dags list-import-errors",
    ]:
        assert term in content


def test_validation_set_runbook_documents_oracle_and_no_train_use() -> None:
    content = Path("docs/validation-set.md").read_text(encoding="utf-8")

    for term in [
        "validation_set_v001",
        "oracle_verdict=conforme",
        "oracle_verdict=defective",
        "human_sophie",
        "bootstrap, calibration et replay",
    ]:
        assert term in content
