"""Path helpers for Streamlit pages running locally or in the /app image."""

from __future__ import annotations

import os
from pathlib import Path


def default_repo_root(page_file: str | os.PathLike[str]) -> Path:
    """Resolve the repository root without assuming a fixed parent depth."""
    env_root = os.environ.get("IQA_REPO_ROOT")
    if env_root:
        return Path(env_root).resolve()

    current = Path(page_file).resolve()
    for candidate in (current.parent, *current.parents):
        if (candidate / "data").exists() and (candidate / "deploy").exists():
            return candidate
        if candidate.name == "iqa-mlops":
            return candidate

    app_root = Path("/app")
    if app_root.exists():
        return app_root
    return current.parent
