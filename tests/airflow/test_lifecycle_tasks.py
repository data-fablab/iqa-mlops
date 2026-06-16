"""Tests for IQA lifecycle DAG tasks (IQA2_KEN11).

Each task is independently testable and returns context for downstream tasks.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from iqa.dags.lifecycle_tasks import (
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

        with patch("iqa.dags.lifecycle_tasks.iter_manifest_image_samples", return_value=[]), \
             patch("iqa.dags.lifecycle_tasks.build_candidate_dataset") as mock_build:
            from iqa.datasets.candidate_builder import CandidateDataset
            mock_build.return_value = CandidateDataset(
                version="v001", sample_count=42, filtered_count=5, output_manifest=out_manifest
            )

            result = task_dataset(**context)

            assert result["manifest_path"] == str(out_manifest)
            assert result["dataset_version"] == "v001"
            assert result["sample_count"] == 42
            mock_build.assert_called_once()


class TestTrainTask:
    """Train Feature AE model."""

    def test_train_task_pulls_manifest_from_xcom(self, tmp_path: Path) -> None:
        """Train task reads manifest_path from ti.xcom_pull(task_ids='dataset')."""
        mock_ti = MagicMock()
        mock_ti.xcom_pull.return_value = {
            "manifest_path": str(tmp_path / "candidate.csv"),
            "dataset_version": "v001",
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

    def test_train_task_uses_mlflow_logging_and_returns_run_id(self, tmp_path: Path) -> None:
        """Train task calls train_feature_ae_with_mlflow_logging and returns run_id."""
        mock_ti = MagicMock()
        mock_ti.xcom_pull.return_value = {
            "manifest_path": str(tmp_path / "candidate.csv"),
            "dataset_version": "v001",
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

            result = task_eval(**context)

            mock_ti.xcom_pull.assert_called_once_with(task_ids="train")
            assert result["recall"] == 1.0
            assert result["ap"] == 0.87


class TestGatesTask:
    """Check promotion gates."""

    def test_gates_task_pulls_metrics_from_eval_xcom(self) -> None:
        """Gates task reads metrics from ti.xcom_pull(task_ids='eval')."""
        mock_ti = MagicMock()
        mock_ti.xcom_pull.return_value = {
            "recall": 0.5, "ap": 0.87, "orange_rate": 0.05, "latency_ms": 800.0
        }
        context = {"ti": mock_ti}

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

            result = task_gates(ti=mock_ti)

            assert result["all_passed"] is True


class TestMLflowTask:
    """Register model in MLflow."""

    def test_mlflow_task_pulls_run_id_from_train_xcom(self) -> None:
        """MLflow task reads run_id from ti.xcom_pull(task_ids='train')."""
        mock_ti = MagicMock()
        mock_ti.xcom_pull.return_value = {"run_id": "abc123def456", "checkpoint": "/tmp/ck.pt"}
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
        context = {"params": {"scenario_id": "drift_domain_extension"}}

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
