from __future__ import annotations

import pytest

import iqa.ingestion as ingestion
from iqa.ingestion import IngestedImage, PieceEvent, build_piece_event


def test_ingestion_package_exports_minimal_contract() -> None:
    assert ingestion.__all__ == [
        "IngestedImage",
        "IngestionSource",
        "PieceEvent",
        "build_piece_event",
    ]


@pytest.mark.parametrize("source", ["historical_replay", "production_ingest"])
def test_piece_event_accepts_replay_and_production_sources(source: str) -> None:
    image = IngestedImage(
        image_id="img-001",
        image_uri="s3://iqa-ingested-images/2026/06/12/img-001.jpg",
        sha256="abc123",
        view_key="top",
    )

    event = build_piece_event(
        piece_event_id="piece-001",
        source=source,  # type: ignore[arg-type]
        lot_id="lot-001",
        source_class="Casting_class1",
        images=[image],
    )

    assert isinstance(event, PieceEvent)
    assert event.source == source
    assert event.images == (image,)


def test_piece_event_requires_at_least_one_image() -> None:
    with pytest.raises(ValueError, match="at least one ingested image"):
        build_piece_event(
            piece_event_id="piece-001",
            source="historical_replay",
            lot_id="lot-001",
            source_class="Casting_class1",
            images=[],
        )
