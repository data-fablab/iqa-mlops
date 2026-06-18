"""Tests for the shared boundary-script helpers (scripts/airflow_contracts.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.airflow_contracts import load_yaml_config, str2bool


@pytest.mark.unit
@pytest.mark.parametrize("value", ["true", "True", "1", "yes", "on", " TRUE "])
def test_str2bool_accepts_truthy_argv_strings(value: str) -> None:
    assert str2bool(value) is True


@pytest.mark.unit
@pytest.mark.parametrize("value", ["false", "0", "no", "off", "", "maybe"])
def test_str2bool_treats_everything_else_as_false(value: str) -> None:
    assert str2bool(value) is False


@pytest.mark.unit
def test_str2bool_passes_through_real_booleans() -> None:
    assert str2bool(True) is True
    assert str2bool(False) is False


@pytest.mark.unit
def test_load_yaml_config_returns_empty_dict_for_a_missing_file(tmp_path: Path) -> None:
    assert load_yaml_config(tmp_path / "absent.yaml") == {}


@pytest.mark.unit
def test_load_yaml_config_returns_empty_dict_for_an_empty_file(tmp_path: Path) -> None:
    config = tmp_path / "empty.yaml"
    config.write_text("", encoding="utf-8")
    assert load_yaml_config(config) == {}


@pytest.mark.unit
def test_load_yaml_config_parses_a_mapping(tmp_path: Path) -> None:
    config = tmp_path / "thresholds.yaml"
    config.write_text("quality:\n  roi_fail_rate_critical: 0.1\n", encoding="utf-8")
    assert load_yaml_config(config) == {"quality": {"roi_fail_rate_critical": 0.1}}
