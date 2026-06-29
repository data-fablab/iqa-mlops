"""Contract for the /heatmap proxy that streams MinIO objects for the review UIs.

Streamlit's ``st.image`` and a Grafana table cell cannot read an ``s3://`` URI;
the dashboards only persist that logical URI. ``GET /heatmap?uri=...`` resolves it
through the same visual object store the inference path writes to and streams the
bytes back as ``image/png`` so the heatmap actually renders.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from iqa.api import main
from iqa.storage.object_store import InMemoryObjectStore
from iqa.storage.uris import IQA_BUCKETS


def _store_with(uri_key: str, data: bytes) -> InMemoryObjectStore:
    return InMemoryObjectStore({(IQA_BUCKETS["heatmaps"], uri_key): data})


def test_heatmap_streams_object_bytes_as_png(monkeypatch: pytest.MonkeyPatch) -> None:
    key = "lots/demo/lot-1/piece-1_img_heatmap.png"
    monkeypatch.setattr(main, "create_visual_object_store", lambda: _store_with(key, b"PNGBYTES"))

    response = main.heatmap(uri=f"s3://{IQA_BUCKETS['heatmaps']}/{key}")

    assert response.media_type == "image/png"
    assert response.body == b"PNGBYTES"


def test_heatmap_missing_object_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "create_visual_object_store", lambda: InMemoryObjectStore())

    with pytest.raises(HTTPException) as excinfo:
        main.heatmap(uri=f"s3://{IQA_BUCKETS['heatmaps']}/missing.png")
    assert excinfo.value.status_code == 404


def test_heatmap_rejects_non_visual_bucket() -> None:
    with pytest.raises(HTTPException) as excinfo:
        main.heatmap(uri="s3://iqa-models/secret.pt")
    assert excinfo.value.status_code == 403


def test_heatmap_rejects_non_s3_uri() -> None:
    with pytest.raises(HTTPException) as excinfo:
        main.heatmap(uri="https://example.com/x.png")
    assert excinfo.value.status_code == 400
