from __future__ import annotations

import pytest

from iqa.runtime import GpuBusyError, gpu_lock, read_gpu_lock_holder


@pytest.fixture(autouse=True)
def _isolated_lock_path(tmp_path, monkeypatch):
    monkeypatch.setenv("IQA_GPU_LOCK_PATH", str(tmp_path / "gpu.lock"))


def test_gpu_lock_records_owner_while_held() -> None:
    with gpu_lock(owner="iqa-trainer") as path:
        assert "iqa-trainer" in read_gpu_lock_holder(path)


def test_second_acquire_is_refused_when_held() -> None:
    with gpu_lock(owner="iqa-inference-demo"):
        with pytest.raises(GpuBusyError, match="iqa-inference-demo"):
            with gpu_lock(owner="iqa-trainer"):
                pass


def test_lock_is_reusable_after_release() -> None:
    with gpu_lock(owner="iqa-trainer"):
        pass
    # No GpuBusyError: the lock was released on context exit.
    with gpu_lock(owner="iqa-trainer") as path:
        assert path.exists()
