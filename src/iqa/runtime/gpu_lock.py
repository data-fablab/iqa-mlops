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
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows
    fcntl = None
    import msvcrt
else:  # pragma: no cover - exercised on POSIX
    msvcrt = None


DEFAULT_GPU_LOCK_PATH = "/tmp/iqa-gpu.lock"
GPU_LOCK_PATH_ENV = "IQA_GPU_LOCK_PATH"
WINDOWS_LOCK_OFFSET = 1_048_576


class GpuBusyError(RuntimeError):
    """Raised when the GPU lock is already held by another process."""


def _lock_file(fd: int, *, blocking: bool) -> None:
    if fcntl is not None:
        flags = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
        fcntl.flock(fd, flags)
        return

    assert msvcrt is not None
    os.lseek(fd, WINDOWS_LOCK_OFFSET, os.SEEK_SET)
    mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
    msvcrt.locking(fd, mode, 1)


def _unlock_file(fd: int) -> None:
    if fcntl is not None:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return

    assert msvcrt is not None
    os.lseek(fd, WINDOWS_LOCK_OFFSET, os.SEEK_SET)
    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)


def _owner_sidecar(path: Path) -> Path:
    return path.with_name(f"{path.name}.owner")


def gpu_lock_path() -> Path:
    """Resolve the lock file path from ``IQA_GPU_LOCK_PATH`` (with a default)."""

    return Path(os.environ.get(GPU_LOCK_PATH_ENV, DEFAULT_GPU_LOCK_PATH))


def read_gpu_lock_holder(path: Path | None = None) -> str:
    """Return the human-readable owner recorded in the lock file, if any."""

    target = path or gpu_lock_path()
    try:
        return target.read_text(encoding="utf-8").strip()
    except OSError:
        if msvcrt is not None:
            try:
                return _owner_sidecar(target).read_text(encoding="utf-8").strip()
            except OSError:
                return ""
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
    locked = False
    try:
        try:
            _lock_file(fd, blocking=blocking)
            locked = True
        except OSError as exc:
            busy_errnos = (errno.EACCES, errno.EAGAIN, getattr(errno, "EDEADLOCK", 36))
            if exc.errno in busy_errnos:
                holder = read_gpu_lock_holder(path) or "another process"
                raise GpuBusyError(
                    f"GPU lock held by {holder}; {owner} refused to run concurrently"
                ) from exc
            raise
        os.ftruncate(fd, 0)
        owner_record = f"{owner} pid={os.getpid()}\n"
        os.write(fd, owner_record.encode())
        os.fsync(fd)
        if msvcrt is not None:
            _owner_sidecar(path).write_text(owner_record, encoding="utf-8")
        yield path
    finally:
        try:
            if locked:
                _unlock_file(fd)
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
