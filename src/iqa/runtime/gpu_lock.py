"""Cross-process GPU lock for the single-GPU IQA server.

The IQA server has one GPU (RTX 3060). Inference (live demo) and training must
never run on it at the same time. The Airflow ``iqa_gpu`` pool (``slots=1``)
already serializes GPU tasks *inside* Airflow; this advisory file lock extends
that guarantee to any process on the host -- e.g. a manual ``iqa-run-lifecycle``
launched during a live inference demo.

Acquire is non-blocking by default: a trainer that finds the lock already held
fails fast (``GpuBusyError``) instead of competing for VRAM with the demo. The
lock file is shared between the inference and trainer containers through a named
Docker volume mounted at ``IQA_GPU_LOCK_PATH``.
"""

from __future__ import annotations

import errno
import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


DEFAULT_GPU_LOCK_PATH = "/tmp/iqa-gpu.lock"
GPU_LOCK_PATH_ENV = "IQA_GPU_LOCK_PATH"


class GpuBusyError(RuntimeError):
    """Raised when the GPU lock is already held by another process."""


def gpu_lock_path() -> Path:
    """Resolve the lock file path from ``IQA_GPU_LOCK_PATH`` (with a default)."""

    return Path(os.environ.get(GPU_LOCK_PATH_ENV, DEFAULT_GPU_LOCK_PATH))


def read_gpu_lock_holder(path: Path | None = None) -> str:
    """Return the human-readable owner recorded in the lock file, if any."""

    target = path or gpu_lock_path()
    try:
        return target.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


@contextmanager
def gpu_lock(*, owner: str, blocking: bool = False) -> Iterator[Path]:
    """Hold an exclusive GPU lock for the duration of the context.

    Parameters
    ----------
    owner:
        Identifier written into the lock file (e.g. ``"iqa-trainer"``) so a
        refused process can report who is holding the GPU.
    blocking:
        When ``False`` (default), raise :class:`GpuBusyError` immediately if the
        lock is held. When ``True``, wait until the lock becomes available.
    """

    path = gpu_lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    flags = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
    try:
        try:
            fcntl.flock(fd, flags)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                holder = read_gpu_lock_holder(path) or "another process"
                raise GpuBusyError(
                    f"GPU lock held by {holder}; {owner} refused to run concurrently"
                ) from exc
            raise
        os.ftruncate(fd, 0)
        os.write(fd, f"{owner} pid={os.getpid()}\n".encode())
        os.fsync(fd)
        yield path
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


__all__ = [
    "DEFAULT_GPU_LOCK_PATH",
    "GPU_LOCK_PATH_ENV",
    "GpuBusyError",
    "gpu_lock",
    "gpu_lock_path",
    "read_gpu_lock_holder",
]
