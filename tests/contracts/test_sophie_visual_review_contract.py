from __future__ import annotations

from pathlib import Path


def test_sophie_review_station_mentions_replay_visual_artifacts() -> None:
    page = Path("deploy/streamlit/pages/2_Review_Sophie.py").read_text(encoding="utf-8")

    for expected in [
        "default_repo_root",
        "IQA_SOPHIE_REPLAY_RUN_DIR",
        "events.jsonl",
        "image_path",
        "heatmap_path",
        "heatmap_uri",
        "roi_mask_uri",
        "Defaut suivant",
        "Conforme suivant",
        "human_sophie",
        "display-only",
        "Exporter feedback Sophie JSONL",
    ]:
        assert expected in page


def test_sophie_review_station_keeps_api_history_as_secondary_view() -> None:
    page = Path("deploy/streamlit/pages/2_Review_Sophie.py").read_text(encoding="utf-8")

    assert "File replay" in page
    assert "Historique API" in page
    assert "/predictions" in page
    assert "/feedback" in page


def test_streamlit_container_can_read_replay_cache_and_raw_images() -> None:
    compose = Path("deploy/docker-compose.yml").read_text(encoding="utf-8")

    assert "IQA_REPO_ROOT: /workspace" in compose
    assert "..:/workspace:ro" in compose


def test_streamlit_pages_do_not_assume_fixed_parent_depth() -> None:
    marc = Path("deploy/streamlit/pages/1_Dashboard_Marc.py").read_text(encoding="utf-8")
    sophie = Path("deploy/streamlit/pages/2_Review_Sophie.py").read_text(encoding="utf-8")

    assert "parents[3]" not in marc
    assert "parents[3]" not in sophie
    assert "default_repo_root(__file__)" in marc
    assert "default_repo_root(__file__)" in sophie
