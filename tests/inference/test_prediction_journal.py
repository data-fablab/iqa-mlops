"""Tests for the JSONL prediction journal seam (Issue 8)."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from iqa.inference.prediction_journal import (
    JOURNAL_FIELDS,
    append_journal,
    journal_entry,
)

pytestmark = pytest.mark.unit


def _sample_prediction(**overrides: object) -> dict:
    base = {
        "piece_event_id": "pe_001",
        "scenario_id": "drift_domain_extension",
        "source_class": "Casting_class1",
        "image_uri": "file:///data/img.jpg",
        "score": 0.025,
        "decision": "Vert",
        "feature_ae_version": "rd_feature_ae_class1_baseline",
        "domain_drift_score": 2.78,
        "domain_regime": "in_domain",
    }
    base.update(overrides)
    return base


class TestJournalEntry:
    def test_contains_all_required_fields(self) -> None:
        entry = journal_entry(_sample_prediction())
        for field in JOURNAL_FIELDS:
            assert field in entry, f"missing field: {field}"

    def test_ts_is_iso_utc(self) -> None:
        entry = journal_entry(_sample_prediction())
        assert entry["ts"].endswith("+00:00") or entry["ts"].endswith("Z")

    def test_preserves_values(self) -> None:
        entry = journal_entry(_sample_prediction(score=0.042, decision="Orange"))
        assert entry["score"] == 0.042
        assert entry["decision"] == "Orange"
        assert entry["piece_event_id"] == "pe_001"

    def test_missing_fields_become_none(self) -> None:
        entry = journal_entry({"piece_event_id": "pe_002"})
        assert entry["source_class"] is None
        assert entry["domain_drift_score"] is None


class TestAppendJournal:
    def test_appends_one_line_per_call(self, tmp_path) -> None:
        path = tmp_path / "journal.jsonl"
        append_journal(_sample_prediction(), path=path)
        append_journal(_sample_prediction(piece_event_id="pe_002"), path=path)
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

    def test_each_line_is_valid_json(self, tmp_path) -> None:
        path = tmp_path / "journal.jsonl"
        append_journal(_sample_prediction(), path=path)
        line = path.read_text(encoding="utf-8").strip()
        parsed = json.loads(line)
        assert parsed["piece_event_id"] == "pe_001"

    def test_creates_parent_dirs(self, tmp_path) -> None:
        path = tmp_path / "sub" / "deep" / "journal.jsonl"
        assert append_journal(_sample_prediction(), path=path)
        assert path.exists()

    def test_graceful_degradation_on_write_failure(self, tmp_path) -> None:
        path = tmp_path / "journal.jsonl"
        with patch("pathlib.Path.open", side_effect=PermissionError("simulated")):
            result = append_journal(_sample_prediction(), path=path)
        assert result is False

    def test_returns_true_on_success(self, tmp_path) -> None:
        path = tmp_path / "journal.jsonl"
        assert append_journal(_sample_prediction(), path=path) is True

    def test_includes_domain_drift_fields(self, tmp_path) -> None:
        path = tmp_path / "journal.jsonl"
        append_journal(
            _sample_prediction(domain_drift_score=4.22, domain_regime="out_of_domain"),
            path=path,
        )
        parsed = json.loads(path.read_text(encoding="utf-8").strip())
        assert parsed["domain_drift_score"] == 4.22
        assert parsed["domain_regime"] == "out_of_domain"
