"""Shared helpers for lightweight Airflow boundary scripts."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def read_csv_rows(path: Path, *, label: str) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def stable_unique(values: Iterable[str | None]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def csv_manifest_summary(path: Path, *, label: str) -> dict[str, Any]:
    rows = read_csv_rows(path, label=label)
    fieldnames = list(rows[0].keys()) if rows else []
    return {
        "path": str(path),
        "row_count": len(rows),
        "field_count": len(fieldnames),
        "dataset_versions": stable_unique(row.get("dataset_version") for row in rows),
        "manifest_versions": stable_unique(row.get("manifest_version") for row in rows),
        "scenario_ids": stable_unique(row.get("scenario_id") for row in rows),
        "source_classes": stable_unique(row.get("source_class") for row in rows),
    }


def print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


__all__ = ["csv_manifest_summary", "print_json", "read_csv_rows", "stable_unique"]
