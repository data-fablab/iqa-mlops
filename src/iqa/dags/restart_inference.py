"""Restart inference container after artifact refresh (Issue 24).

Runs on the **scheduler** (the only host with the Docker socket mounted).
Sequence: restart container → poll ``/health`` → ``/predict`` warmup → done.

A corrupted active artifact causes the warmup ``/predict`` to fail, which
makes the task fail (relaunched) instead of leaving a broken container live.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_CONTAINER_NAME = "deploy-iqa-inference-1"
DEFAULT_INFERENCE_URL = "http://iqa-inference:8100"
HEALTH_POLL_INTERVAL = 2.0
HEALTH_POLL_TIMEOUT = 60.0
WARMUP_TIMEOUT = 30.0


@dataclass
class RestartResult:
    status: str
    container: str
    health_ok: bool = False
    warmup_ok: bool = False
    reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "container": self.container,
            "health_ok": self.health_ok,
            "warmup_ok": self.warmup_ok,
            "reason": self.reason,
        }


def _restart_container(container_name: str) -> None:
    """Restart a container via the Docker SDK."""
    import docker

    client = docker.from_env()
    container = client.containers.get(container_name)
    logger.info("restarting container %s", container_name)
    container.restart(timeout=30)
    logger.info("container %s restarted", container_name)


def _poll_health(base_url: str, *, timeout: float = HEALTH_POLL_TIMEOUT) -> bool:
    """Poll /health until OK or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(f"{base_url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
                if resp.status == 200:
                    logger.info("health check passed")
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(HEALTH_POLL_INTERVAL)
    logger.error("health check timed out after %.0fs", timeout)
    return False


def _warmup_predict(base_url: str, warmup_image: str, *, timeout: float = WARMUP_TIMEOUT) -> bool:
    """POST /predict with a known-good image to force lazy model loading."""
    payload = json.dumps({"image_path": warmup_image}).encode("utf-8")
    try:
        req = urllib.request.Request(
            f"{base_url}/predict",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            result = json.loads(resp.read().decode("utf-8"))
            logger.info("warmup predict succeeded: %s", result.get("status", "ok"))
            return True
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        logger.error("warmup predict failed: %s", exc)
        return False


def restart_inference(
    container_name: str | None = None,
    inference_url: str | None = None,
    warmup_image: str | None = None,
    *,
    restart_fn=None,
    health_fn=None,
    warmup_fn=None,
) -> RestartResult:
    """Restart + validate the inference container.

    Injectable ``restart_fn``/``health_fn``/``warmup_fn`` for unit testing.
    """
    name = container_name or os.environ.get("IQA_INFERENCE_CONTAINER", DEFAULT_CONTAINER_NAME)
    url = inference_url or os.environ.get("IQA_INFERENCE_URL", DEFAULT_INFERENCE_URL)
    image = warmup_image or os.environ.get(
        "IQA_WARMUP_IMAGE",
        "data/raw/hss-iad/Casting_class1/train/good/cast_ok_0_0.jpeg",
    )

    do_restart = restart_fn or _restart_container
    do_health = health_fn or _poll_health
    do_warmup = warmup_fn or _warmup_predict

    try:
        do_restart(name)
    except Exception as exc:
        return RestartResult(status="error", container=name, reason=f"restart_failed: {exc}")

    health_ok = do_health(url)
    if not health_ok:
        return RestartResult(status="error", container=name, reason="health_timeout")

    warmup_ok = do_warmup(url, image)
    if not warmup_ok:
        return RestartResult(
            status="error", container=name, health_ok=True,
            reason="warmup_predict_failed",
        )

    return RestartResult(
        status="restarted", container=name,
        health_ok=True, warmup_ok=True,
    )


def task_restart_inference(**context) -> dict:
    """Airflow task entry point."""
    params = context.get("params", {})
    result = restart_inference(
        container_name=params.get("inference_container"),
        inference_url=params.get("inference_url"),
        warmup_image=params.get("warmup_image"),
    )
    if result.status == "error":
        raise RuntimeError(f"restart_inference failed: {result.reason}")
    return result.to_dict()
