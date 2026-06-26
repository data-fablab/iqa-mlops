"""Append-only JSONL prediction journal (Issue 8, ADR 0010 decision 5).

Passive collection: one line per ``/predict`` with the fields the future
resolver C (feedback-store retrain) needs to join a label to a prediction.
Written on the hot path without DB coupling (file on a mounted volume).

A write failure never breaks ``/predict`` — the journal degrades silently.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_JOURNAL_PATH = "/var/log/iqa/predictions.jsonl"

JOURNAL_FIELDS = (
    "ts",
    "piece_event_id",
    "scenario_id",
    "source_class",
    "image_uri",
    "score",
    "decision",
    "feature_ae_version",
    "domain_drift_score",
    "domain_regime",
)

_lock = threading.Lock()


def journal_path() -> Path:
    return Path(os.environ.get("IQA_PREDICTION_JOURNAL", DEFAULT_JOURNAL_PATH))


def journal_entry(prediction: dict[str, Any]) -> dict[str, Any]:
    """Build a journal line from a prediction dict (pure, no I/O)."""
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "piece_event_id": prediction.get("piece_event_id"),
        "scenario_id": prediction.get("scenario_id"),
        "source_class": prediction.get("source_class"),
        "image_uri": prediction.get("image_uri"),
        "score": prediction.get("score"),
        "decision": prediction.get("decision"),
        "feature_ae_version": prediction.get("feature_ae_version"),
        "domain_drift_score": prediction.get("domain_drift_score"),
        "domain_regime": prediction.get("domain_regime"),
    }


def append_journal(prediction: dict[str, Any], *, path: Path | None = None) -> bool:
    """Append one JSONL line. Returns True on success, False on degradation."""
    target = path or journal_path()
    entry = journal_entry(prediction)
    line = json.dumps(entry, separators=(",", ":"), default=str) + "\n"
    try:
        with _lock:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as fh:
                fh.write(line)
        return True
    except Exception:  # noqa: BLE001 - never break /predict
        logger.warning("prediction journal write failed for %s", target, exc_info=True)
        return False


__all__ = [
    "DEFAULT_JOURNAL_PATH",
    "JOURNAL_FIELDS",
    "append_journal",
    "journal_entry",
    "journal_path",
]
