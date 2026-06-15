"""MLflow run logging with full traceability for model training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import mlflow
    import mlflow.pytorch
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

from iqa.training.feature_ae import FeatureAETrainingConfig
from iqa.training.feature_ae_evaluation import EvaluationReport


class MLflowRunLogger:
    """Logger for MLflow runs with complete traceability."""

    def __init__(
        self,
        run_name: str,
        scenario_id: str,
        tracking_uri: str | None = None,
    ) -> None:
        """Initialize MLflow run logger.

        Args:
            run_name: Name for the MLflow run
            scenario_id: Scenario identifier for tags
            tracking_uri: MLflow tracking URI (local file:// or remote http://)
        """
        if not HAS_MLFLOW:
            raise ImportError("MLflow is required for MLflowRunLogger")

        self.run_name = run_name
        self.scenario_id = scenario_id
        self.tracking_uri = tracking_uri
        self.run: Any = None
        self._run_id: str | None = None

        # Configure MLflow backend
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

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
            "tile_stride": config.tile_stride,
            "preprocessing_mode": config.preprocessing_mode,
            "loss": config.loss,
            "scenario_id": config.scenario_id,
            "dataset_version": config.dataset_version,
            "candidate_version": config.candidate_version,
            "roi_model_version": config.roi_model_version,
            "feature_ae_version": config.feature_ae_version,
        }
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log training metrics at each step.

        Args:
            metrics: Dictionary of metric_name -> value
            step: Step/epoch number
        """
        for name, value in metrics.items():
            mlflow.log_metric(name, value, step=step)

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
            mlflow.log_artifact(str(checkpoint_path), artifact_path="model")

        if eval_report_path and eval_report_path.exists():
            mlflow.log_artifact(str(eval_report_path), artifact_path="reports")

    def set_tags(
        self,
        git_commit: str,
        dataset_version: str,
        scenario_id: str,
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
            "scenario_id": scenario_id,
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
    )

    try:
        # Log configuration and tags
        logger.log_config(config)
        logger.set_tags(
            git_commit=git_commit,
            dataset_version=config.dataset_version,
            scenario_id=config.scenario_id,
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
        if checkpoint_path.exists():
            logger.log_artifacts(checkpoint_path=checkpoint_path)

        # Log evaluation report if it exists
        eval_best_path = run_dir / "metric_eval_best.json"
        if eval_best_path.exists():
            logger.log_artifacts(eval_report_path=eval_best_path)

        return result
    finally:
        # Always end the run
        logger.end_run()


__all__ = ["MLflowRunLogger", "train_feature_ae_with_mlflow_logging"]
