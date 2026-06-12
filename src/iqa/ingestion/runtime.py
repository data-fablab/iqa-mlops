"""Pure helpers for building ingestion events."""

from __future__ import annotations

from collections.abc import Iterable

from iqa.ingestion.schemas import IngestedImage, IngestionSource, PieceEvent


def build_piece_event(
    *,
    piece_event_id: str,
    source: IngestionSource,
    lot_id: str,
    source_class: str,
    images: Iterable[IngestedImage],
) -> PieceEvent:
    image_tuple = tuple(images)
    if not image_tuple:
        raise ValueError("A piece event must reference at least one ingested image.")

    return PieceEvent(
        piece_event_id=piece_event_id,
        source=source,
        lot_id=lot_id,
        source_class=source_class,
        images=image_tuple,
    )


__all__ = ["build_piece_event"]
