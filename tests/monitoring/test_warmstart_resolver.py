"""Tests for warm-start checkpoint resolution (Issue 25)."""

from __future__ import annotations

import pytest

from iqa.monitoring.warmstart_resolver import resolve_warmstart_checkpoint

pytestmark = pytest.mark.unit


def test_resolve_from_yaml(tmp_path):
    config = tmp_path / "warmstart.yaml"
    config.write_text(
        "warmstart_checkpoints:\n"
        "  Casting_class2: /models/class2/checkpoint.pt\n"
        "  Casting_class3: /models/class3/checkpoint.pt\n",
        encoding="utf-8",
    )
    result = resolve_warmstart_checkpoint("Casting_class2", config_path=config)
    assert result == "/models/class2/checkpoint.pt"


def test_resolve_returns_none_when_class_not_mapped(tmp_path):
    config = tmp_path / "warmstart.yaml"
    config.write_text(
        "warmstart_checkpoints:\n"
        "  Casting_class2: /models/class2/checkpoint.pt\n",
        encoding="utf-8",
    )
    result = resolve_warmstart_checkpoint("Casting_class99", config_path=config)
    assert result is None


def test_resolve_returns_none_when_file_absent(tmp_path):
    result = resolve_warmstart_checkpoint(
        "Casting_class2", config_path=tmp_path / "nonexistent.yaml"
    )
    assert result is None


def test_resolve_returns_none_on_empty_yaml(tmp_path):
    config = tmp_path / "warmstart.yaml"
    config.write_text("", encoding="utf-8")
    result = resolve_warmstart_checkpoint("Casting_class2", config_path=config)
    assert result is None


def test_resolve_returns_none_on_invalid_yaml(tmp_path):
    config = tmp_path / "warmstart.yaml"
    config.write_text(": invalid: yaml: [", encoding="utf-8")
    result = resolve_warmstart_checkpoint("Casting_class2", config_path=config)
    assert result is None


def test_resolve_string_coercion(tmp_path):
    config = tmp_path / "warmstart.yaml"
    config.write_text(
        "warmstart_checkpoints:\n  Casting_class2: 42\n",
        encoding="utf-8",
    )
    result = resolve_warmstart_checkpoint("Casting_class2", config_path=config)
    assert result == "42"
