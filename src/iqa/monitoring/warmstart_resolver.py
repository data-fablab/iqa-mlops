"""Demo warm-start checkpoint resolution (Issue 25).

Maps ``triggering_class`` to a pre-baked checkpoint path via a YAML file.
File absent or class not mapped → ``None`` (no override, prod behaviour).
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

DEFAULT_WARMSTART_CONFIG = "configs/demo_warmstart_checkpoints.yaml"


def resolve_warmstart_checkpoint(
    triggering_class: str,
    config_path: str | Path | None = None,
) -> str | None:
    """Return a checkpoint path for ``triggering_class``, or ``None``."""
    path = Path(config_path or DEFAULT_WARMSTART_CONFIG)
    if not path.exists():
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (yaml.YAMLError, OSError) as exc:
        logger.warning("warmstart config unreadable (%s): %s", path, exc)
        return None
    checkpoints = data.get("warmstart_checkpoints", {})
    result = checkpoints.get(triggering_class)
    if result is not None:
        result = str(result)
    return result
