"""FastAPI entrypoints for IQA."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from iqa.api.main import app

__all__ = ["app"]


def __getattr__(name: str):
    # Lazy so that importing a leaf module (e.g. iqa.api.schemas) does not drag
    # in iqa.api.main. This keeps `from iqa.api import app` working while letting
    # sibling packages import iqa.api.schemas at runtime without an import cycle.
    if name == "app":
        from iqa.api.main import app

        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
