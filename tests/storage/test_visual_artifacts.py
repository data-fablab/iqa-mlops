from pathlib import Path

from iqa.storage.object_store import InMemoryObjectStore
from iqa.storage.uris import IQA_BUCKETS, parse_s3_uri
from iqa.storage.visual_artifacts import VisualArtifactContext, publish_heatmap, publish_roi_mask, visual_artifact_key


def test_visual_artifact_key_is_stable_and_scoped() -> None:
    key = visual_artifact_key(
        VisualArtifactContext(
            scenario_id="production_replay_natural",
            lot_id="LOT 001",
            piece_event_id="piece/001",
            image_id="img:001",
        ),
        artifact="heatmap",
    )

    assert key == "lots/production_replay_natural/LOT_001/piece_001_img_001_heatmap.png"


def test_publish_roi_and_heatmap_return_minio_uris(tmp_path: Path) -> None:
    store = InMemoryObjectStore()
    context = VisualArtifactContext("scenario", "lot", "piece", "image")
    artifact = tmp_path / "artifact.png"
    artifact.write_bytes(b"png")

    roi_uri = publish_roi_mask(artifact, context, store=store)
    heatmap_uri = publish_heatmap(artifact, context, store=store)

    assert parse_s3_uri(roi_uri).bucket == IQA_BUCKETS["roi_masks"]
    assert parse_s3_uri(heatmap_uri).bucket == IQA_BUCKETS["heatmaps"]
    assert store.get_bytes(IQA_BUCKETS["roi_masks"], parse_s3_uri(roi_uri).key) == b"png"
    assert store.get_bytes(IQA_BUCKETS["heatmaps"], parse_s3_uri(heatmap_uri).key) == b"png"
