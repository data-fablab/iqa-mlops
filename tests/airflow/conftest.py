"""Shared fixtures for the Airflow boundary-script tests."""

from __future__ import annotations

import json
import sys
from typing import Callable

import pytest


@pytest.fixture
def run_boundary_script(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> Callable[[object, list[str]], dict]:
    """Drive a boundary script through its public interface: argv in, parsed JSON out.

    This is the exact contract Airflow relies on -- a one-shot container invoked
    with templated argv that prints its result (the task XCom) to stdout.
    """

    def _run(module: object, args: list[str]) -> dict:
        monkeypatch.setattr(sys, "argv", args)
        module.main()
        return json.loads(capsys.readouterr().out)

    return _run
