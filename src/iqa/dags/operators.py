"""Airflow operator factory for containerised IQA tasks (ADR 0008).

This module is the single place that decides *how* a task runs as a container.
Today it returns a ``DockerOperator``; the same call site can target a
``KubernetesPodOperator`` later by setting ``IQA_AIRFLOW_BACKEND=k8s`` -- this is
what keeps the door open to Kubernetes (ADR 0002) without touching the DAGs.

Design rule (ADR 0008): **no import of the iqa runtime here**. The Airflow
scheduler imports this module; it must only need Airflow + providers, never the
metier code (torch, pandas, models...). Each container carries its own runtime.

Backend is selected by env var:

- ``IQA_AIRFLOW_BACKEND=docker`` (default) -> ``DockerOperator``
- ``IQA_AIRFLOW_BACKEND=k8s``               -> ``KubernetesPodOperator`` (stub)

Other env knobs (read lazily, so importing this module never requires them):

- ``IQA_DOCKER_URL``       docker socket (default ``unix://var/run/docker.sock``)
- ``IQA_DOCKER_NETWORK``   network the task container joins (services discovery)
- ``IQA_GPU_LOCK_VOLUME``  named volume carrying the single-GPU lock (default ``iqa_gpu_lock``)
- ``IQA_GPU_LOCK_PATH``    lock file path inside the container (default ``/var/run/iqa-gpu/gpu.lock``)
- ``IQA_K8S_NAMESPACE``    namespace for the k8s backend (default ``default``)
"""

from __future__ import annotations

import os
from posixpath import dirname
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime Airflow import.
    from airflow.models import BaseOperator

DEFAULT_DOCKER_URL = "unix://var/run/docker.sock"
DEFAULT_GPU_LOCK_VOLUME = "iqa_gpu_lock"
DEFAULT_GPU_LOCK_PATH = "/var/run/iqa-gpu/gpu.lock"


def _backend() -> str:
    return os.environ.get("IQA_AIRFLOW_BACKEND", "docker").strip().lower()


def _normalise_command(command: str | list[str] | None) -> list[str] | None:
    """Return the command as a token list (k8s wants ``cmds`` as a list)."""
    if command is None:
        return None
    if isinstance(command, str):
        return command.split()
    return list(command)


def make_container_task(
    *,
    task_id: str,
    image: str,
    command: str | list[str] | None = None,
    env: dict[str, str] | None = None,
    pool: str | None = None,
    gpu_lock: bool = False,
    **kwargs: Any,
) -> BaseOperator:
    """Build the Airflow operator that runs ``image`` as a one-shot container.

    The container exit code propagates to Airflow: a non-zero exit fails the
    task (both backends raise on non-zero, which is the Airflow default).

    Set ``gpu_lock=True`` on GPU-bound tasks: the single-GPU lock volume is
    mounted into the container so the file lock is shared with the inference
    service (one holder at a time). Pair it with ``pool="iqa_gpu"`` (slots=1).

    Extra keyword arguments are forwarded to the underlying operator, so DAGs
    keep full control (retries, ``execution_timeout``, ``trigger_rule``, ...).
    """
    backend = _backend()
    if backend == "docker":
        return _make_docker_task(
            task_id=task_id, image=image, command=command, env=env, pool=pool,
            gpu_lock=gpu_lock, **kwargs
        )
    if backend == "k8s":
        return _make_k8s_task(
            task_id=task_id, image=image, command=command, env=env, pool=pool,
            gpu_lock=gpu_lock, **kwargs
        )
    raise ValueError(
        f"Unknown IQA_AIRFLOW_BACKEND={backend!r} (expected 'docker' or 'k8s')"
    )


def _make_docker_task(
    *,
    task_id: str,
    image: str,
    command: str | list[str] | None,
    env: dict[str, str] | None,
    pool: str | None,
    gpu_lock: bool,
    **kwargs: Any,
) -> BaseOperator:
    from airflow.providers.docker.operators.docker import DockerOperator
    from docker.types import Mount

    environment = dict(env or {})
    params: dict[str, Any] = {
        "task_id": task_id,
        "image": image,
        "command": command,
        "environment": environment,
        "docker_url": os.environ.get("IQA_DOCKER_URL", DEFAULT_DOCKER_URL),
        # auto-remove the container once done; never leak stopped containers.
        "auto_remove": "success",
        # we manage our own data plane (MinIO/PG/MLflow); no host tmp mount.
        "mount_tmp_dir": False,
    }
    if pool is not None:
        params["pool"] = pool
    network = os.environ.get("IQA_DOCKER_NETWORK")
    if network:
        params["network_mode"] = network
    if gpu_lock:
        lock_path = os.environ.get("IQA_GPU_LOCK_PATH", DEFAULT_GPU_LOCK_PATH)
        environment.setdefault("IQA_GPU_LOCK_PATH", lock_path)
        params["mounts"] = [
            Mount(
                source=os.environ.get("IQA_GPU_LOCK_VOLUME", DEFAULT_GPU_LOCK_VOLUME),
                target=dirname(lock_path),
                type="volume",
            )
        ]
    params.update(kwargs)
    return DockerOperator(**params)


def _make_k8s_task(
    *,
    task_id: str,
    image: str,
    command: str | list[str] | None,
    env: dict[str, str] | None,
    pool: str | None,
    gpu_lock: bool,
    **kwargs: Any,
) -> BaseOperator:
    """Kubernetes backend -- stub, not exercised yet (ADR 0002 escape hatch).

    Kept import-light and documented so the migration to KPO is a config flip,
    not a rewrite. ``command`` maps to the pod ``cmds`` (token list); our images
    define no ENTRYPOINT, so the console script is the command itself.

    TODO (k8s migration): ``gpu_lock`` should map to a node GPU resource request
    (``resources``) and/or a ``ReadWriteMany`` PVC, not a docker named volume.
    """
    del gpu_lock  # not yet wired for the k8s backend (see TODO above)
    from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
    from kubernetes.client import models as k8s

    params: dict[str, Any] = {
        "task_id": task_id,
        "image": image,
        "cmds": _normalise_command(command),
        "env_vars": [k8s.V1EnvVar(name=k, value=v) for k, v in (env or {}).items()],
        "namespace": os.environ.get("IQA_K8S_NAMESPACE", "default"),
        "name": task_id.replace("_", "-"),
        # surface the pod exit code as the task result, then clean up.
        "get_logs": True,
        "is_delete_operator_pod": True,
    }
    if pool is not None:
        params["pool"] = pool
    params.update(kwargs)
    return KubernetesPodOperator(**params)


__all__ = ["make_container_task"]
