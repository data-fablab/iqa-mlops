"""Minimal ingestion schemas shared by replay and production sources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


IngestionSource = Literal["historical_replay", "production_ingest"]


@dataclass(frozen=True)
class IngestedImage:
    image_id: str
    image_uri: str
    sha256: str
    view_key: str


@dataclass(frozen=True)
class PieceEvent:
    piece_event_id: str
    source: IngestionSource
    lot_id: str
    source_class: str
    images: tuple[IngestedImage, ...]
    event_time: str = ""
    recorded_at: str = ""

    @property
    def is_simulated(self) -> bool:
        return self.source == "historical_replay"


__all__ = ["IngestedImage", "IngestionSource", "PieceEvent"]
