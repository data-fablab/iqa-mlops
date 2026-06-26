from __future__ import annotations

import json
import sys

import pytest

from iqa.metadata.repository import MemoryMetadataRepository
from scripts import collect_lifecycle_signal


def test_collector_requires_postgres_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        collect_lifecycle_signal,
        "metadata_backend",
        lambda: "memory",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "iqa-collect-lifecycle-signal",
            "--scenario-id",
            "production_replay_natural",
        ],
    )

    with pytest.raises(RuntimeError, match="IQA_METADATA_BACKEND=postgres"):
        collect_lifecycle_signal.main()


def test_collector_prints_persisted_decision(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = MemoryMetadataRepository()

    monkeypatch.setattr(
        collect_lifecycle_signal,
        "metadata_backend",
        lambda: "postgres",
    )
    monkeypatch.setattr(
        collect_lifecycle_signal,
        "create_metadata_repository",
        lambda: repository,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "iqa-collect-lifecycle-signal",
            "--scenario-id",
            "production_replay_natural",
        ],
    )

    collect_lifecycle_signal.main()

    output = capsys.readouterr().out
    assert '"service": "iqa-lifecycle-signal-collector"' in output
    assert '"trigger_lifecycle": false' in output

    xcom_payload = json.loads(output.strip().splitlines()[-1])
    assert xcom_payload["service"] == "iqa-lifecycle-signal-collector"
    assert xcom_payload["trigger_lifecycle"] is False
    assert len(repository.list_lifecycle_trigger_events()) == 1
