"""Tests for restart_inference task (Issue 24)."""

from __future__ import annotations

import pytest

from iqa.dags.restart_inference import RestartResult, restart_inference

pytestmark = pytest.mark.unit


def test_restart_full_sequence():
    restart_calls = []

    def mock_restart(name):
        restart_calls.append(name)

    result = restart_inference(
        container_name="test-container",
        inference_url="http://localhost:8100",
        warmup_image="test.jpg",
        restart_fn=mock_restart,
        health_fn=lambda url: True,
        warmup_fn=lambda url, img: True,
    )

    assert result.status == "restarted"
    assert result.health_ok is True
    assert result.warmup_ok is True
    assert restart_calls == ["test-container"]


def test_restart_fails_on_restart_error():
    def failing_restart(name):
        raise RuntimeError("docker offline")

    result = restart_inference(
        container_name="test",
        restart_fn=failing_restart,
        health_fn=lambda url: True,
        warmup_fn=lambda url, img: True,
    )

    assert result.status == "error"
    assert "restart_failed" in result.reason


def test_restart_fails_on_health_timeout():
    result = restart_inference(
        container_name="test",
        restart_fn=lambda name: None,
        health_fn=lambda url: False,
        warmup_fn=lambda url, img: True,
    )

    assert result.status == "error"
    assert result.reason == "health_timeout"


def test_restart_fails_on_warmup_failure():
    result = restart_inference(
        container_name="test",
        restart_fn=lambda name: None,
        health_fn=lambda url: True,
        warmup_fn=lambda url, img: False,
    )

    assert result.status == "error"
    assert result.reason == "warmup_predict_failed"
    assert result.health_ok is True


def test_restart_result_round_trips():
    r = RestartResult(status="restarted", container="c", health_ok=True, warmup_ok=True)
    d = r.to_dict()
    assert d["status"] == "restarted"
    assert d["container"] == "c"
