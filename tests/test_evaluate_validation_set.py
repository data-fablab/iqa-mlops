"""Tests for Feature AE evaluation on frozen validation_set_v001."""

from __future__ import annotations

import json
from pathlib import Path


from iqa.training.feature_ae_evaluation import EvaluationReport, evaluate_on_validation_set_v001


class TestEvaluateValidationSet:
    """Test Feature AE evaluation on validation_set_v001."""

    def test_evaluation_report_has_required_fields(self, tmp_path: Path) -> None:
        """EvaluationReport has all required metric fields."""
        report = EvaluationReport(
            model_version="v2",
            average_precision=0.95,
            recall=0.92,
            orange_rate=0.03,
            latency_ms=150.5,
            sample_count=100,
        )

        assert report.model_version == "v2"
        assert report.average_precision == 0.95
        assert report.recall == 0.92
        assert report.orange_rate == 0.03
        assert report.latency_ms == 150.5
        assert report.sample_count == 100

    def test_evaluation_report_to_dict(self, tmp_path: Path) -> None:
        """EvaluationReport can serialize to dict."""
        report = EvaluationReport(
            model_version="v2",
            average_precision=0.95,
            recall=0.92,
            orange_rate=0.03,
            latency_ms=150.5,
            sample_count=100,
        )

        data = report.to_dict()

        assert isinstance(data, dict)
        assert data["model_version"] == "v2"
        assert data["average_precision"] == 0.95
        assert data["recall"] == 0.92
        assert data["orange_rate"] == 0.03
        assert data["latency_ms"] == 150.5

    def test_evaluation_report_to_json(self, tmp_path: Path) -> None:
        """EvaluationReport can serialize to JSON."""
        report = EvaluationReport(
            model_version="v2",
            average_precision=0.95,
            recall=0.92,
            orange_rate=0.03,
            latency_ms=150.5,
            sample_count=100,
        )

        json_str = report.to_json()

        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert data["model_version"] == "v2"

    def test_evaluate_on_validation_set_returns_report(
        self,
        tmp_path: Path,
        synthetic_feature_ae_checkpoint: Path,
        synthetic_validation_manifest: Path,
        synthetic_image_root: Path,
    ) -> None:
        """evaluate_on_validation_set_v001 returns EvaluationReport."""
        report = evaluate_on_validation_set_v001(
            checkpoint_path=synthetic_feature_ae_checkpoint,
            manifest_path=synthetic_validation_manifest,
            image_root=synthetic_image_root,
            output_dir=tmp_path,
            model_version="v2",
        )

        assert isinstance(report, EvaluationReport)
        assert report.model_version == "v2"

    def test_evaluation_includes_all_metrics(
        self,
        tmp_path: Path,
        synthetic_feature_ae_checkpoint: Path,
        synthetic_validation_manifest: Path,
        synthetic_image_root: Path,
    ) -> None:
        """Evaluation report includes AP, recall, Orange rate, latency."""
        report = evaluate_on_validation_set_v001(
            checkpoint_path=synthetic_feature_ae_checkpoint,
            manifest_path=synthetic_validation_manifest,
            image_root=synthetic_image_root,
            output_dir=tmp_path,
            model_version="v2",
        )

        assert report.average_precision >= 0.0
        assert 0.0 <= report.recall <= 1.0
        assert 0.0 <= report.orange_rate <= 1.0
        # Latency is measured during inference, so it must be strictly positive
        # (a 0.0 would mean the metric fell back to its default instead of being wired).
        assert report.latency_ms > 0.0
        assert report.sample_count > 0

    def test_evaluation_saves_report(
        self,
        tmp_path: Path,
        synthetic_feature_ae_checkpoint: Path,
        synthetic_validation_manifest: Path,
        synthetic_image_root: Path,
    ) -> None:
        """Evaluation saves JSON report to output_dir."""
        output_dir = tmp_path / "eval_results"
        report = evaluate_on_validation_set_v001(
            checkpoint_path=synthetic_feature_ae_checkpoint,
            manifest_path=synthetic_validation_manifest,
            image_root=synthetic_image_root,
            output_dir=output_dir,
            model_version="v2",
        )

        report_path = output_dir / "evaluation_report.json"
        assert report_path.exists()

        saved_data = json.loads(report_path.read_text())
        assert saved_data["model_version"] == "v2"
        assert saved_data["average_precision"] == report.average_precision
