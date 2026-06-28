"""MLflow run logging with full traceability for model training."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

try:
    import mlflow
    import mlflow.pyfunc
    import mlflow.pytorch
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

from iqa.models.artifacts import model_manifest_path
from iqa.training.feature_ae import FeatureAETrainingConfig
from iqa.training.feature_ae_contracts import (
    FEATURE_AE_PREPROCESSING_CONTRACT_VERSION,
    canonical_feature_ae_preprocessing_dict,
)
from iqa.training.feature_ae_evaluation import EvaluationReport

# Training runs (params + datasets + checkpoint) land here so they are visible
# next to the model-quality eval runs, instead of MLflow's "Default" experiment.
# Matches monitoring.model_metrics.MODEL_QUALITY_EXPERIMENT; override per deploy
# with MLFLOW_EXPERIMENT_NAME. Routing here is safe for the candidate-vs-prod gate:
# it filters on the ``stage`` tag, which training runs do not set.
DEFAULT_EXPERIMENT_NAME = "iqa-model-quality"


class FeatureAEReferencePyfuncModel(mlflow.pyfunc.PythonModel if HAS_MLFLOW else object):
    """Traceability wrapper for Feature-AE model versions in MLflow."""

    def predict(self, context: Any, model_input: list[Any], params: dict[str, Any] | None = None) -> list[Any]:
        raise NotImplementedError(
            "Feature-AE MLflow model artifacts are for lineage and registry traceability; "
            "use the IQA runtime loader for GPU inference."
        )


class MLflowRunLogger:
    """Logger for MLflow runs with complete traceability."""

    def __init__(
        self,
        run_name: str,
        scenario_id: str,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
    ) -> None:
        """Initialize MLflow run logger.

        Args:
            run_name: Name for the MLflow run
            scenario_id: Scenario identifier for tags
            tracking_uri: MLflow tracking URI (local file:// or remote http://)
            experiment_name: Experiment to log into. Defaults to
                ``MLFLOW_EXPERIMENT_NAME`` env or ``iqa-model-quality`` so runs are
                discoverable instead of falling into MLflow's "Default" experiment.
        """
        if not HAS_MLFLOW:
            raise ImportError("MLflow is required for MLflowRunLogger")

        self.run_name = run_name
        self.scenario_id = scenario_id
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name or os.environ.get(
            "MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME
        )
        self.run: Any = None
        self._run_id: str | None = None

        # Configure MLflow backend
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Pin the experiment so training runs (with their dataset lineage) are not
        # scattered into "Default".
        mlflow.set_experiment(self.experiment_name)

        # Start the run
        self.run = mlflow.start_run(run_name=run_name)
        self._run_id = mlflow.active_run().info.run_id

    def log_config(self, config: FeatureAETrainingConfig) -> None:
        """Log training parameters from config.

        Args:
            config: FeatureAETrainingConfig object
        """
        # Log numeric and string params that MLflow accepts
        params = {
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "cosine_weight": config.cosine_weight,
            "roi_loss_weight": config.roi_loss_weight,
            "background_loss_weight": config.background_loss_weight,
            "lr_scheduler": config.lr_scheduler,
            "lr_patience": config.lr_patience,
            "lr_factor": config.lr_factor,
            "early_stopping_patience": config.early_stopping_patience,
            "metric_early_stopping_patience": config.metric_early_stopping_patience,
            "metric_eval_every_epochs": config.metric_eval_every_epochs,
            "checkpoint_selection_policy": "business_metric_only" if config.metric_eval_manifest_path else "loss_only_no_metric_eval",
            "layer_score_mode": "sqrt_l2_plus_cosine",
            "layer_normalization": "good_p99",
            "preprocessing_contract_version": FEATURE_AE_PREPROCESSING_CONTRACT_VERSION,
            "tile_stride": config.tile_stride,
            "preprocessing_mode": config.preprocessing_mode,
            "loss": config.loss,
            "scenario_id": config.scenario_id,
            "dataset_version": config.dataset_version,
            "manifest_version": config.manifest_version,
            "candidate_version": config.candidate_version,
            "roi_model_version": config.roi_model_version,
            "feature_ae_version": config.feature_ae_version,
        }
        mlflow.log_params(params)

    def log_datasets(self, config: FeatureAETrainingConfig) -> dict[str, bool]:
        """Log MLflow Dataset inputs for training and metric evaluation manifests."""
        logged = {
            "training": _log_manifest_dataset(
                manifest_path=config.manifest_path,
                name=config.dataset_version or config.manifest_path.stem,
                context="training",
            ),
            "metric_eval": False,
        }
        if config.metric_eval_manifest_path:
            logged["metric_eval"] = _log_manifest_dataset(
                manifest_path=config.metric_eval_manifest_path,
                name=config.validation_set_id or config.metric_eval_manifest_path.stem,
                context="metric_eval",
            )
        return logged

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log training metrics at each step.

        Args:
            metrics: Dictionary of metric_name -> value
            step: Step/epoch number
        """
        for name, value in metrics.items():
            mlflow.log_metric(name, value, step=step)

    def log_business_metrics(self, eval_best_path: Path) -> dict[str, float]:
        """Log the business metrics from ``metric_eval_best.json`` as MLflow metrics.

        The evaluator already computes image_ap, image_auroc, pixel_ap and the
        low-FPR AUPIMO (``pixel_aupimo_1e-5_1e-3``) per epoch and records the best
        in ``metric_eval_best.json``, but the wrapper only attached it as an
        artifact -- so the run showed loss curves and no business metrics. This
        promotes each ``{value: ...}`` entry to a first-class MLflow metric (plus a
        friendly ``aupimo`` alias) so they are charted next to train/val loss.
        """
        try:
            payload = json.loads(eval_best_path.read_text(encoding="utf-8-sig"))
        except (json.JSONDecodeError, OSError):
            return {}
        logged: dict[str, float] = {}
        for key, entry in payload.items():
            if isinstance(entry, dict) and isinstance(entry.get("value"), (int, float)):
                value = float(entry["value"])
                mlflow.log_metric(key, value)
                logged[key] = value
        aupimo = logged.get("pixel_aupimo_1e-5_1e-3")
        if aupimo is not None:
            mlflow.log_metric("aupimo", aupimo)
        return logged

    def log_evaluation_metrics(self, eval_report: EvaluationReport) -> None:
        """Log evaluation metrics from EvaluationReport.

        Args:
            eval_report: EvaluationReport dataclass
        """
        metrics = {
            "average_precision": eval_report.average_precision,
            "recall": eval_report.recall,
            "orange_rate": eval_report.orange_rate,
            "latency_ms": eval_report.latency_ms,
            "sample_count": eval_report.sample_count,
        }
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

    def log_artifacts(
        self,
        checkpoint_path: Path | None = None,
        eval_report_path: Path | None = None,
    ) -> None:
        """Log model artifacts.

        Args:
            checkpoint_path: Path to model checkpoint file
            eval_report_path: Path to evaluation report file
        """
        if checkpoint_path and checkpoint_path.exists():
            mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")

        if eval_report_path and eval_report_path.exists():
            mlflow.log_artifact(str(eval_report_path), artifact_path="reports")

    def log_feature_ae_model(self, config: FeatureAETrainingConfig, checkpoint_path: Path) -> bool:
        """Log a real MLflow Model wrapper for the Feature-AE checkpoint."""
        if not checkpoint_path.exists():
            return False
        with tempfile.TemporaryDirectory(prefix="iqa_feature_ae_mlflow_") as tmp:
            tmp_path = Path(tmp)
            contract_path = tmp_path / "score_contract.json"
            contract_path.write_text(
                json.dumps(canonical_feature_ae_preprocessing_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            artifacts = {
                "checkpoint": str(checkpoint_path),
                "score_contract": str(contract_path),
            }
            model_version = config.candidate_version or config.feature_ae_version
            if model_version:
                source_manifest = model_manifest_path(model_version)
                if source_manifest.exists():
                    manifest_artifact = tmp_path / "model_manifest.json"
                    shutil.copy2(source_manifest, manifest_artifact)
                    artifacts["model_manifest"] = str(manifest_artifact)
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=FeatureAEReferencePyfuncModel(),
                artifacts=artifacts,
                metadata={
                    "model_type": "feature_ae_reference",
                    "scenario_id": config.scenario_id,
                    "candidate_version": config.candidate_version,
                    "score_contract_version": FEATURE_AE_PREPROCESSING_CONTRACT_VERSION,
                    "checkpoint_filename": checkpoint_path.name,
                },
            )
        return True

    def set_tags(
        self,
        git_commit: str,
        dataset_version: str,
        scenario_id: str,
        manifest_version: str = "",
        model_version: str = "",
        candidate_version: str = "",
        roi_model_version: str = "",
        feature_ae_version: str = "",
        preprocessing_contract_version: str = FEATURE_AE_PREPROCESSING_CONTRACT_VERSION,
    ) -> None:
        """Set traceability tags.

        Args:
            git_commit: Git commit hash
            dataset_version: Dataset version identifier
            scenario_id: Scenario identifier
        """
        tags = {
            "git_commit": git_commit,
            "dataset_version": dataset_version,
            "dataset_snapshot_id": dataset_version,
            "manifest_version": manifest_version,
            "scenario_id": scenario_id,
            "score_contract_version": preprocessing_contract_version,
            "model_version": model_version or candidate_version,
            "candidate_version": candidate_version,
            "roi_model_version": roi_model_version,
            "feature_ae_version": feature_ae_version,
            "preprocessing_contract_version": preprocessing_contract_version,
        }
        mlflow.set_tags(tags)

    def end_run(self) -> str:
        """End the MLflow run.

        Returns:
            The run_id of the completed run
        """
        mlflow.end_run()
        return self._run_id or ""


def train_feature_ae_with_mlflow_logging(
    config: FeatureAETrainingConfig,
    git_commit: str,
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
) -> dict[str, Any]:
    """Train Feature-AE with complete MLflow logging.

    Wraps train_feature_ae() and logs params, metrics, artifacts, and tags to MLflow.

    Args:
        config: FeatureAETrainingConfig
        git_commit: Git commit hash for traceability
        tracking_uri: MLflow tracking URI (defaults to env var MLFLOW_TRACKING_URI)

    Returns:
        Training results dict (same as train_feature_ae)
    """
    import csv

    from iqa.training.feature_ae import train_feature_ae

    # Initialize MLflow logger
    logger = MLflowRunLogger(
        run_name=config.run_name or f"feature_ae_{config.scenario_id}",
        scenario_id=config.scenario_id,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
    )

    try:
        # Log configuration and tags
        logger.log_config(config)
        dataset_logging = logger.log_datasets(config)
        logger.set_tags(
            git_commit=git_commit,
            dataset_version=config.dataset_version,
            scenario_id=config.scenario_id,
            manifest_version=config.manifest_version,
            model_version=config.candidate_version,
            candidate_version=config.candidate_version,
            roi_model_version=config.roi_model_version,
            feature_ae_version=config.feature_ae_version,
            preprocessing_contract_version=FEATURE_AE_PREPROCESSING_CONTRACT_VERSION,
        )

        # Train the model
        result = train_feature_ae(config)

        # Log training metrics from loss history
        run_dir = Path(result["run_dir"])
        loss_history_path = run_dir / "loss_history.csv"
        if loss_history_path.exists():
            with loss_history_path.open() as f:
                reader = csv.DictReader(f)
                for row in reader:
                    metrics = {
                        k: float(v)
                        for k, v in row.items()
                        if k != "epoch" and v
                    }
                    step = int(row["epoch"]) if row.get("epoch") else 0
                    if metrics:
                        logger.log_metrics(metrics, step=step)

        # Log artifacts from the run directory
        checkpoint_path = Path(result["checkpoint"])

        # Log final checkpoint
        model_logged = False
        if checkpoint_path.exists():
            logger.log_artifacts(checkpoint_path=checkpoint_path)
            model_logged = logger.log_feature_ae_model(config, checkpoint_path)

        # Log evaluation report if it exists -- as an artifact AND as first-class
        # MLflow metrics (image_ap, image_auroc, pixel_ap, AUPIMO), so the run shows
        # the business metrics next to the loss curves.
        eval_best_path = run_dir / "metric_eval_best.json"
        business_metrics: dict[str, float] = {}
        if eval_best_path.exists():
            logger.log_artifacts(eval_report_path=eval_best_path)
            business_metrics = logger.log_business_metrics(eval_best_path)
        eval_history_path = run_dir / "metric_eval_history.json"
        if eval_history_path.exists():
            logger.log_artifacts(eval_report_path=eval_history_path)

        result["run_id"] = logger._run_id or ""
        result["mlflow_business_metrics_logged"] = sorted(business_metrics)
        result["mlflow_dataset_logged"] = any(dataset_logging.values())
        result["mlflow_training_dataset_logged"] = dataset_logging["training"]
        result["mlflow_metric_eval_dataset_logged"] = dataset_logging["metric_eval"]
        result["mlflow_model_logged"] = model_logged
        return result
    finally:
        # Always end the run
        logger.end_run()


def _log_manifest_dataset(*, manifest_path: Path, name: str, context: str) -> bool:
    if not manifest_path.exists():
        return False
    try:
        import pandas as pd
    except ImportError:
        return False
    frame = pd.read_csv(manifest_path)
    try:
        dataset = mlflow.data.from_pandas(frame, source=str(manifest_path), name=name)
    except TypeError:
        dataset = mlflow.data.from_pandas(frame, name=name)
    mlflow.log_input(dataset, context=context)
    return True


__all__ = ["FeatureAEReferencePyfuncModel", "MLflowRunLogger", "train_feature_ae_with_mlflow_logging"]
