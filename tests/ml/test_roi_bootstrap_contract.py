from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from iqa.roi import RoiPredictionArtifact
from iqa.roi.bootstrap import BOOTSTRAP_SOURCE, generate_bootstrap_roi_predictions


@dataclass(frozen=True)
class _FakePrediction:
    roi_ratio: float = 0.42
    roi_quality_status: str = "ok"


def test_roi_prediction_artifact_contract() -> None:
    artifact = RoiPredictionArtifact(
        piece_event_id="piece_event_001",
        image_id="image_001",
        image_uri="Casting_class1/train/good/sample.jpg",
        roi_mask_uri="s3://iqa-roi-masks/bootstrap_v001/image_001_roi.png",
        roi_model_version="roi_segmenter_v001_fixed",
        roi_ratio=0.42,
        roi_quality_status="ok",
        source=BOOTSTRAP_SOURCE,
        scenario_id="bootstrap_v001",
        dataset_version="feature_ae_good_v001_bootstrap",
    )

    assert artifact.to_dict()["roi_mask_uri"].startswith("s3://iqa-roi-masks/")
    assert artifact.to_dict()["roi_quality_status"] == "ok"


def test_generate_bootstrap_roi_predictions_writes_processed_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image_root = tmp_path / "images"
    image_path = image_root / "Casting_class1/train/good/sample.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"fake image bytes")
    manifest = tmp_path / "feature_ae_bootstrap_events.csv"
    manifest.write_text(
        "event_id,image_ids,relative_paths,label,is_defective,bootstrap_dataset_version,bootstrap_role\n"
        "piece_event_001,image_001,Casting_class1/train/good/sample.jpg,good,false,feature_ae_good_v001_bootstrap,train_normal_piece_b_minimal\n",
        encoding="utf-8",
    )
    calls: list[Path] = []

    def fake_predict_roi_image(*args, **kwargs):
        calls.append(Path(kwargs["output_mask"]))
        kwargs["output_mask"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["output_mask"].write_bytes(b"fake mask")
        return _FakePrediction()

    monkeypatch.setattr("iqa.roi.bootstrap.predict_roi_image", fake_predict_roi_image)

    output_dir = tmp_path / "data/processed/roi/bootstrap_v001"
    artifacts = generate_bootstrap_roi_predictions(
        manifest_path=manifest,
        image_root=image_root,
        checkpoint_path=tmp_path / "checkpoint.pt",
        output_dir=output_dir,
        roi_model_version="roi_segmenter_v001_fixed",
        device="cpu",
    )

    assert len(artifacts) == 1
    assert artifacts[0].source == BOOTSTRAP_SOURCE
    assert artifacts[0].roi_mask_uri.endswith("piece_event_001_image_001_roi.png")
    assert output_dir.joinpath("roi_predictions.csv").is_file()
    assert calls[0].parts[-3:] == ("bootstrap_v001", "masks", "piece_event_001_image_001_roi.png")


def test_generate_bootstrap_roi_predictions_rejects_reports_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not reports"):
        generate_bootstrap_roi_predictions(
            manifest_path=tmp_path / "missing.csv",
            image_root=tmp_path,
            checkpoint_path=tmp_path / "checkpoint.pt",
            output_dir=tmp_path / "reports/smoke_roi",
            roi_model_version="roi_segmenter_v001_fixed",
        )


def test_generate_bootstrap_roi_predictions_rejects_non_good_rows(tmp_path: Path) -> None:
    manifest = tmp_path / "feature_ae_bootstrap_events.csv"
    manifest.write_text(
        "event_id,image_ids,relative_paths,label,is_defective,bootstrap_dataset_version,bootstrap_role\n"
        "piece_event_bad,image_bad,sample.jpg,defective,true,feature_ae_good_v001_bootstrap,train_normal\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not good-only"):
        generate_bootstrap_roi_predictions(
            manifest_path=manifest,
            image_root=tmp_path,
            checkpoint_path=tmp_path / "checkpoint.pt",
            output_dir=tmp_path / "data/processed/roi/bootstrap_v001",
            roi_model_version="roi_segmenter_v001_fixed",
        )
