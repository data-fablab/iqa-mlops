"""Tests for IQA lifecycle DAG tasks (IQA2_KEN11).

Each task is independently testable and returns context for downstream tasks.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from iqa.dags.lifecycle_tasks import (
    task_lifecycle_decision,
    task_dataset,
    task_eval,
    task_gates,
    task_mlflow,
    task_promotion,
    task_reload,
    task_train,
)

# Task callables are plain Python; they run without an Airflow runtime.
pytestmark = pytest.mark.unit


# Airflow context type for testing
class AirflowContext(dict):
    """Simulates Airflow task context."""

    pass


class TestDatasetTask:
    """Prepare dataset for training."""

    def test_dataset_task_returns_manifest_path_and_version(self, tmp_path: Path) -> None:
        """Dataset task builds candidate dataset from params and returns manifest_path."""
        manifest = tmp_path / "source.csv"
        manifest.write_text("image_id,relative_path,source_class,label,is_defective,split_set,event_id,scenario_id,dataset_version\n")
        out_manifest = tmp_path / "candidate.csv"

        context = {
            "params": {
                "manifest_path": str(manifest),
                "image_root": str(tmp_path),
                "output_manifest": str(out_manifest),
                "candidate_version": "v001",
            }
        }

        roi_lookup = SimpleNamespace(status={"img_1": "ok"})
        with patch("iqa.dags.lifecycle_tasks.iter_manifest_image_samples", return_value=[]), \
             patch("iqa.dags.lifecycle_tasks.load_roi_mask_lookup", return_value=roi_lookup), \
             patch("iqa.dags.lifecycle_tasks.build_candidate_dataset") as mock_build:
            from iqa.datasets.candidate_builder import CandidateDataset
            mock_build.return_value = CandidateDataset(
                version="v001", sample_count=42, filtered_count=5, output_manifest=out_manifest
            )

            result = task_dataset(**context)

            assert result["manifest_path"] == str(out_manifest)
            assert result["dataset_version"] == "v001"
            assert result["manifest_version"] == "v001_manifest_v001"
            assert result["sample_count"] == 42
            assert result["roi_status_count"] == 1
            mock_build.assert_called_once()
            assert mock_build.call_args.kwargs["roi_status"] == {"img_1": "ok"}

    def test_dataset_task_warns_without_roi_predictions(self, tmp_path: Path) -> None:
        """Dataset task keeps MVP behavior but reports when ROI status is absent."""
        manifest = tmp_path / "source.csv"
        manifest.write_text("image_id,relative_path,source_class,label,is_defective,split_set,event_id,scenario_id,dataset_version\n")
        out_manifest = tmp_path / "candidate.csv"
        context = {
            "params": {
                "manifest_path": str(manifest),
                "image_root": str(tmp_path),
                "output_manifest": str(out_manifest),
            }
        }

        with patch("iqa.dags.lifecycle_tasks.iter_manifest_image_samples", return_value=[]), \
             patch("iqa.dags.lifecycle_tasks.load_roi_mask_lookup") as mock_roi_lookup, \
             patch("iqa.dags.lifecycle_tasks.build_candidate_dataset") as mock_build:
            from iqa.datasets.candidate_builder import CandidateDataset

            mock_roi_lookup.return_value = SimpleNamespace(status={})
            mock_build.return_value = CandidateDataset(
                version="v001", sample_count=0, filtered_count=0, output_manifest=out_manifest
            )

            result = task_dataset(**context)

            assert "warning" in result

    def test_dataset_task_skips_when_lifecycle_decision_is_not_triggered(self, tmp_path: Path) -> None:
        """Dataset task does not build when the lifecycle decision says no trigger."""
        mock_ti = MagicMock()
        mock_ti.xcom_pull.return_value = {
            "trigger_lifecycle": False,
            "trigger_reason": "natural_waiting_for_50_oracle_conformes",
        }
        context = {
            "ti": mock_ti,
            "params": {
                "manifest_path": str(tmp_path / "source.csv"),
                "image_root": str(tmp_path),
                "output_manifest": str(tmp_path / "candidate.csv"),
            },
        }

        with patch("iqa.dags.lifecycle_tasks.build_candidate_dataset") as mock_build:
            result = task_dataset(**context)

            mock_build.assert_not_called()
            assert result["status"] == "skipped"
            assert result["reason"] == "lifecycle_decision_not_triggered"


class TestLifecycleDecisionTask:
    """Evaluate data-event lifecycle decisions before training."""

    def test_lifecycle_decision_task_triggers_natural_v002_at_50_conformes(self) -> None:
        result = task_lifecycle_decision(
            params={
                "scenario_id": "production_replay_natural",
                "conforming_validated_count": 50,
                "drift_confirmed": False,
            }
        )

        assert result["trigger_lifecycle"] is True
        assert result["candidate_dataset_version"] == "feature_ae_good_v002"

    def test_lifecycle_decision_task_triggers_drift_v003_on_confirmed_drift(self) -> None:
        result = task_lifecycle_decision(
            params={
                "scenario_id": "drift_domain_extension",
                "conforming_validated_count": 0,
                "drift_confirmed": True,
            }
        )

        assert result["trigger_lifecycle"] is True
        assert result["candidate_dataset_version"] == "feature_ae_good_v003"


class TestTrainTask:
    """Train Feature AE model."""

    def test_train_task_pulls_manifest_from_xcom(self, tmp_path: Path) -> None:
        """Train task reads manifest_path from ti.xcom_pull(task_ids='dataset')."""
        mock_ti = MagicMock()
        mock_ti.xcom_pull.return_value = {
            "manifest_path": str(tmp_path / "candidate.csv"),
            "dataset_version": "v001",
            "manifest_version": "v001_manifest_v001",
        }
        context = {
            "ti": mock_ti,
            "params": {
                "scenario_id": "production_replay_natural",
                "image_root": str(tmp_path),
                "output_checkpoint": str(tmp_path / "checkpoint.pt"),
            },
        }

        with patch("iqa.dags.lifecycle_tasks.train_feature_ae_with_mlflow_logging") as mock_train:
            mock_train.return_value = {
                "run_id": "abc123",
                "checkpoint": str(tmp_path / "checkpoint.pt"),
                "run_dir": str(tmp_path),
            }

            task_train(**context)

            mock_ti.xcom_pull.assert_called_once_with(task_ids="dataset")
            config = mock_train.call_args.args[0]
            assert config.roi_model_version == "roi_segmenter_v001_fixed"
            assert config.feature_ae_version == "rd_feature_ae_gated_v001_bootstrap"
            assert config.dataset_version == "v001"
            assert config.manifest_version == "v001_manifest_v001"

    def test_train_task_uses_mlflow_logging_and_returns_run_id(self, tmp_path: Path) -> None:
        """Train task calls train_feature_ae_with_mlflow_logging and returns run_id."""
        mock_ti = MagicMock()
        mock_ti.xcom_pull.return_value = {
            "manifest_path": str(tmp_path / "candidate.csv"),
            "dataset_version": "v001",
            "manifest_version": "v001_manifest_v001",
        }
        context = {
            "ti": mock_ti,
            "params": {
                "scenario_id": "production_replay_natural",
                "image_root": str(tmp_path),
                "output_checkpoint": str(tmp_path / "checkpoint.pt"),
            },
        }

        with patch("iqa.dags.lifecycle_tasks.train_feature_ae_with_mlflow_logging") as mock_train:
            mock_train.return_value = {
                "run_id": "abc123",
                "checkpoint": str(tmp_path / "checkpoint.pt"),
                "run_dir": str(tmp_path),
            }

            result = task_train(**context)

            mock_train.assert_called_once()
            assert result["run_id"] == "abc123"
            assert "checkpoint" in result
            assert result["dataset_version"] == "v001"
            assert result["manifest_version"] == "v001_manifest_v001"


class TestEvalTask:
    """Evaluate Feature AE on validation set."""

    def test_eval_task_pulls_checkpoint_from_xcom(self, tmp_path: Path) -> None:
        """Eval task reads checkpoint path from ti.xcom_pull(task_ids='train')."""
        mock_ti = MagicMock()
        mock_ti.xcom_pull.return_value = {
            "checkpoint": str(tmp_path / "checkpoint.pt"),
            "run_id": "abc123",
        }
        context = {
            "ti": mock_ti,
            "params": {
                "manifest_path": str(tmp_path / "val.csv"),
                "image_root": str(tmp_path),
            },
        }

        with patch("iqa.dags.lifecycle_tasks.evaluate_feature_ae_checkpoint") as mock_eval:
            mock_eval.return_value = {
                "metrics": {
                    "image_recall": 1.0, "image_ap": 0.87,
                    "orange_rate": 0.05, "latency_ms": 800.0,
                    "false_negatives": 0,
                },
                "images": [],
            }

            with patch("iqa.dags.lifecycle_tasks.log_model_quality_metrics", return_value="quality_run_1"):
                result = task_eval(**context)

            mock_ti.xcom_pull.assert_called_once_with(task_ids="train")
            assert result["recall"] == 1.0
            assert result["ap"] == 0.87

    def test_eval_task_logs_business_metrics_to_model_quality_experiment(self, tmp_path: Path) -> None:
        """Eval task logs the 4 business metrics to iqa-model-quality with model_version/stage tags."""
        mock_ti = MagicMock()
        mock_ti.xcom_pull.return_value = {
            "checkpoint": str(tmp_path / "checkpoint.pt"),
            "run_id": "abc123",
            "dataset_version": "feature_ae_good_v002",
        }
        context = {
            "ti": mock_ti,
            "params": {
                "manifest_path": str(tmp_path / "val.csv"),
                "image_root": str(tmp_path),
                "candidate_version": "rd_feature_ae_gated_natural_cycle_001",
                "mlflow_tracking_uri": "http://mlflow:5000",
            },
        }

        eval_metrics = {
            "pixel_aupimo_1e-5_1e-3": 0.42,
            "pixel_ap": 0.61,
            "image_ap": 0.87,
            "image_auroc": 0.93,
            "image_recall": 1.0,
            "orange_rate": 0.05,
            "latency_ms": 800.0,
            "false_negatives": 0,
        }
        with patch("iqa.dags.lifecycle_tasks.evaluate_feature_ae_checkpoint") as mock_eval, \
             patch("iqa.dags.lifecycle_tasks.log_model_quality_metrics") as mock_log:
            mock_eval.return_value = {"metrics": eval_metrics, "images": []}
            mock_log.return_value = "quality_run_42"

            result = task_eval(**context)

            mock_log.assert_called_once()
            logged_metrics = mock_log.call_args.args[0]
            for key in ("pixel_aupimo_1e-5_1e-3", "pixel_ap", "image_ap", "image_auroc"):
                assert logged_metrics[key] == eval_metrics[key]
            call_kwargs = mock_log.call_args.kwargs
            assert call_kwargs["model_version"] == "rd_feature_ae_gated_natural_cycle_001"
            assert call_kwargs["stage"] == "candidate"
            assert call_kwargs["tracking_uri"] == "http://mlflow:5000"
            assert result["model_quality_run_id"] == "quality_run_42"
            assert result["model_version"] == "rd_feature_ae_gated_natural_cycle_001"
            assert result["stage"] == "candidate"

    def test_eval_task_returns_quality_metrics_and_logs_per_class(self, tmp_path: Path) -> None:
        """Eval forwards the 4 business metrics and logs per-class metrics (Issue 4)."""
        mock_ti = MagicMock()
        mock_ti.xcom_pull.return_value = {
            "checkpoint": str(tmp_path / "checkpoint.pt"),
            "run_id": "abc123",
        }
        context = {
            "ti": mock_ti,
            "params": {
                "manifest_path": str(tmp_path / "val.csv"),
                "image_root": str(tmp_path),
                "candidate_version": "cand_v1",
            },
        }
        eval_metrics = {
            "pixel_aupimo_1e-5_1e-3": 0.42,
            "pixel_ap": 0.61,
            "image_ap": 0.87,
            "image_auroc": 0.93,
            "image_recall": 1.0,
            "orange_rate": 0.05,
            "latency_ms": 800.0,
        }
        per_class = {
            "class1": {"image_ap": 0.95, "pixel_ap": 0.7},
            "class2": {"image_ap": 0.80, "pixel_ap": 0.5},
        }
        with patch("iqa.dags.lifecycle_tasks.evaluate_feature_ae_checkpoint") as mock_eval, \
             patch("iqa.dags.lifecycle_tasks.log_model_quality_metrics", return_value="q_run"), \
             patch("iqa.dags.lifecycle_tasks.log_per_class_quality_metrics") as mock_per_class:
            mock_eval.return_value = {
                "metrics": eval_metrics,
                "per_class_metrics": per_class,
                "images": [],
            }

            result = task_eval(**context)

            assert result["quality_metrics"]["pixel_aupimo_1e-5_1e-3"] == 0.42
            assert result["quality_metrics"]["image_ap"] == 0.87
            assert result["per_class_metrics"] == per_class
            mock_per_class.assert_called_once()
            assert mock_per_class.call_args.kwargs["run_id"] == "q_run"

    def test_eval_task_skips_when_train_skipped(self) -> None:
        """Eval task propagates a skip and does not log metrics when training was skipped."""
        mock_ti = MagicMock()
        mock_ti.xcom_pull.return_value = {
            "status": "skipped",
            "reason": "lifecycle_decision_not_triggered",
        }
        context = {"ti": mock_ti, "params": {}}

        with patch("iqa.dags.lifecycle_tasks.log_model_quality_metrics") as mock_log:
            result = task_eval(**context)

            mock_log.assert_not_called()
            assert result["status"] == "skipped"


class TestGatesTask:
    """Check promotion gates."""

    def test_gates_task_pulls_metrics_from_eval_xcom(self) -> None:
        """Gates task reads metrics from ti.xcom_pull(task_ids='eval')."""
        mock_ti = MagicMock()
        mock_ti.xcom_pull.return_value = {
            "recall": 0.5, "ap": 0.87, "orange_rate": 0.05, "latency_ms": 800.0
        }
        context = {
            "ti": mock_ti,
            "params": {"gates_config_path": "configs/promotion_gates.yaml"},
        }

        with patch("iqa.dags.lifecycle_tasks.evaluate_promotion_gates") as mock_gates:
            mock_gates.return_value = {
                "all_passed": False,
                "gates": {"recall": {"passed": False}},
                "rollback_signal": True,
            }

            with pytest.raises(Exception, match="Gates failed"):
                task_gates(**context)

            mock_ti.xcom_pull.assert_called_once_with(task_ids="eval")
            # recall from XCom was passed to the gate, not a default
            call_kwargs = mock_gates.call_args[1]
            assert call_kwargs["candidate_recall"] == 0.5
            assert "gates_config" in call_kwargs

    def test_gates_task_passes_when_all_gates_pass(self) -> None:
        """Gates task succeeds if all gates pass."""
        mock_ti = MagicMock()
        mock_ti.xcom_pull.return_value = {
            "recall": 1.0, "ap": 0.87, "orange_rate": 0.05, "latency_ms": 800.0
        }
        with patch("iqa.dags.lifecycle_tasks.evaluate_promotion_gates") as mock_gates:
            mock_gates.return_value = {
                "all_passed": True,
                "gates": {"recall": {"passed": True}},
                "rollback_signal": False,
            }

            result = task_gates(
                ti=mock_ti,
                params={"gates_config_path": "configs/promotion_gates.yaml"},
            )

            assert result["all_passed"] is True

    def test_gates_task_runs_four_metric_regression_vs_prod_baseline(self) -> None:
        """When candidate quality metrics exist, the gate fetches the prod baseline."""
        mock_ti = MagicMock()
        mock_ti.xcom_pull.return_value = {
            "recall": 1.0, "ap": 0.87, "orange_rate": 0.05, "latency_ms": 800.0,
            "quality_metrics": {
                "pixel_aupimo_1e-5_1e-3": 0.42, "pixel_ap": 0.61,
                "image_ap": 0.87, "image_auroc": 0.93,
            },
        }
        prod_baseline = {
            "pixel_aupimo_1e-5_1e-3": 0.40, "pixel_ap": 0.60,
            "image_ap": 0.85, "image_auroc": 0.92,
        }
        with patch(
            "iqa.dags.lifecycle_tasks.fetch_latest_quality_metrics",
            return_value=prod_baseline,
        ) as mock_fetch:
            result = task_gates(
                ti=mock_ti,
                params={"gates_config_path": "configs/promotion_gates.yaml"},
            )

            mock_fetch.assert_called_once()
            assert result["all_passed"] is True
            quality = result["gates"]["quality_regression"]["verdict"]
            assert quality["decisive_metric"] == "pixel_aupimo_1e-5_1e-3"
            assert quality["all_passed"] is True

    def test_gates_task_blocks_on_quality_regression_vs_prod(self) -> None:
        """A candidate regressing past tolerance on the decisive metric is blocked."""
        mock_ti = MagicMock()
        mock_ti.xcom_pull.return_value = {
            "recall": 1.0, "ap": 0.87, "orange_rate": 0.05, "latency_ms": 800.0,
            "quality_metrics": {
                "pixel_aupimo_1e-5_1e-3": 0.30,  # 0.10 drop vs prod 0.40
                "pixel_ap": 0.61, "image_ap": 0.87, "image_auroc": 0.93,
            },
        }
        prod_baseline = {
            "pixel_aupimo_1e-5_1e-3": 0.40, "pixel_ap": 0.60,
            "image_ap": 0.85, "image_auroc": 0.92,
        }
        with patch(
            "iqa.dags.lifecycle_tasks.fetch_latest_quality_metrics",
            return_value=prod_baseline,
        ):
            with pytest.raises(Exception, match="Gates failed"):
                task_gates(
                    ti=mock_ti,
                    params={"gates_config_path": "configs/promotion_gates.yaml"},
                )


class TestMLflowTask:
    """Register model in MLflow."""

    def test_mlflow_task_pulls_run_id_from_train_xcom(self) -> None:
        """MLflow task reads run_id from ti.xcom_pull(task_ids='train')."""
        mock_ti = MagicMock()
        mock_ti.xcom_pull.return_value = {
            "run_id": "abc123def456",
            "checkpoint": "/tmp/ck.pt",
            "dataset_version": "feature_ae_good_v002",
            "manifest_version": "feature_ae_good_v002_manifest_v001",
        }
        context = {
            "ti": mock_ti,
            "params": {"scenario_id": "production_replay_natural"},
        }

        with patch("iqa.dags.lifecycle_tasks.register_run_to_model") as mock_register:
            mock_register.return_value = {
                "registered_model_name": "feature_ae__production_replay_natural",
                "version": "3",
                "stage": "candidate",
            }

            result = task_mlflow(**context)

            mock_ti.xcom_pull.assert_called_once_with(task_ids="train")
            call_kwargs = mock_register.call_args[1]
            assert call_kwargs["run_id"] == "abc123def456"
            assert result["version"] == "3"
            assert result["dataset_version"] == "feature_ae_good_v002"
            assert result["manifest_version"] == "feature_ae_good_v002_manifest_v001"

    def test_mlflow_task_uses_scenario_id_from_params(self) -> None:
        """MLflow task reads scenario_id from context['params'] (not flat kwargs)."""
        mock_ti = MagicMock()
        mock_ti.xcom_pull.return_value = {"run_id": "xyz789", "checkpoint": "/tmp/ck.pt"}
        context = {
            "ti": mock_ti,
            "params": {"scenario_id": "drift_domain_extension"},
        }

        with patch("iqa.dags.lifecycle_tasks.register_run_to_model") as mock_register:
            mock_register.return_value = {
                "registered_model_name": "feature_ae__drift_domain_extension",
                "version": "1",
                "stage": "candidate",
            }

            task_mlflow(**context)

            call_kwargs = mock_register.call_args[1]
            assert call_kwargs["scenario_id"] == "drift_domain_extension"


class TestPromotionTask:
    """Promote model from candidate to test/prod."""

    def test_promotion_task_pulls_model_ref_from_mlflow_xcom(self) -> None:
        """Promotion task reads registered_model_name and version from mlflow XCom."""
        mock_ti = MagicMock()
        mock_ti.xcom_pull.side_effect = lambda task_ids: {
            "mlflow": {"registered_model_name": "feature_ae__production_replay_natural", "version": "3"},
            "eval": {"recall": 1.0, "ap": 0.87, "orange_rate": 0.05, "latency_ms": 800.0},
        }[task_ids]
        context = {"ti": mock_ti}

        with patch("iqa.dags.lifecycle_tasks.promote_model_with_gates") as mock_promote:
            mock_promote.return_value = {
                "success": True,
                "gates_passed": True,
                "transition": {"success": True},
                "artifacts": {"artifact_uri": "s3://iqa-models/..."},
            }

            result = task_promotion(**context)

            call_kwargs = mock_promote.call_args[1]
            assert call_kwargs["registered_model_name"] == "feature_ae__production_replay_natural"
            assert call_kwargs["version"] == "3"
            assert call_kwargs["target_stage"] == "test"
            assert result["success"] is True

    def test_promotion_task_saves_previous_prod_for_prod_target(self) -> None:
        """Production promotion preserves the current prod alias before transition."""
        mock_ti = MagicMock()
        mock_ti.xcom_pull.side_effect = lambda task_ids: {
            "mlflow": {"registered_model_name": "feature_ae__production_replay_natural", "version": "3"},
            "eval": {"recall": 1.0, "ap": 0.87, "orange_rate": 0.05, "latency_ms": 800.0},
        }[task_ids]

        with patch("iqa.dags.lifecycle_tasks.save_previous_prod_before_promotion") as mock_save, \
             patch("iqa.dags.lifecycle_tasks.promote_model_with_gates") as mock_promote:
            mock_save.return_value = {"success": True, "previous_prod_version": "2"}
            mock_promote.return_value = {
                "success": True,
                "gates_passed": True,
                "transition": {"success": True},
                "artifacts": {"artifact_uri": "s3://iqa-models/..."},
            }

            result = task_promotion(ti=mock_ti, params={"target_stage": "prod"})

            mock_save.assert_called_once_with("feature_ae__production_replay_natural")
            assert mock_promote.call_args.kwargs["target_stage"] == "prod"
            assert result["success"] is True

    def test_promotion_task_blocks_when_gates_fail(self) -> None:
        """Promotion task raises when gates fail."""
        mock_ti = MagicMock()
        mock_ti.xcom_pull.side_effect = lambda task_ids: {
            "mlflow": {"registered_model_name": "feature_ae__production_replay_natural", "version": "3"},
            "eval": {"recall": 0.8, "ap": 0.5, "orange_rate": 0.2, "latency_ms": 2000.0},
        }[task_ids]

        with patch("iqa.dags.lifecycle_tasks.promote_model_with_gates") as mock_promote:
            mock_promote.return_value = {
                "success": False,
                "gates_passed": False,
                "blocked_reasons": ["recall"],
            }

            with pytest.raises(Exception, match="Promotion blocked"):
                task_promotion(ti=mock_ti)


class TestReloadTask:
    """Reload model in inference service."""

    def test_reload_task_reads_scenario_id_from_params(self) -> None:
        """Reload task reads scenario_id from context['params'], not flat kwargs."""
        context = {"params": {"scenario_id": "drift_domain_extension", "target_stage": "prod"}}

        with patch("iqa.dags.lifecycle_tasks.ProdModelLoader") as mock_loader_class:
            mock_loader = MagicMock()
            mock_loader_class.return_value = mock_loader
            mock_loaded = MagicMock()
            mock_loaded.version = "2"
            mock_loaded.artifact_uri = "s3://iqa-models/..."
            mock_loaded.registered_model_name = "feature_ae__drift_domain_extension"
            mock_loader.reload.return_value = mock_loaded

            result = task_reload(**context)

            mock_loader_class.assert_called_once_with("drift_domain_extension")
            assert result["version"] == "2"
            assert "artifact_uri" in result

    def test_reload_task_skips_non_prod_target(self) -> None:
        """Staging/test DAG runs do not reload the current prod model."""
        with patch("iqa.dags.lifecycle_tasks.ProdModelLoader") as mock_loader_class:
            result = task_reload(params={"target_stage": "test"})

            mock_loader_class.assert_not_called()
            assert result["status"] == "skipped"
            assert result["target_stage"] == "test"
