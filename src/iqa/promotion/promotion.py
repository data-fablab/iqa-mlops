"""Model promotion workflow: gate evaluation, MLflow transitions, artifact resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from iqa.promotion.gates import evaluate_promotion_gates


def evaluate_gates_for_promotion(
    registered_model_name: str,
    candidate_metrics: dict[str, float],
    gates_config: dict[str, Any],
    prod_metrics: dict[str, float] | None = None,
    candidate_quality_metrics: dict[str, Any] | None = None,
    prod_quality_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate promotion gates for a candidate model.

    Args:
        registered_model_name: Name of the registered model
        candidate_metrics: Candidate model metrics (recall, ap, orange_rate, latency_ms)
        gates_config: Gate thresholds config
        prod_metrics: Production model metrics (optional, for AP regression check)
        candidate_quality_metrics: Candidate's 4 business metrics (optional). With
            ``prod_quality_metrics`` this enables the 4-metric non-regression gate.
        prod_quality_metrics: Prod baseline's 4 business metrics (optional).

    Returns:
        Decision object with:
        - passed: bool, True if all gates pass
        - blocked: bool, True if any gate fails
        - blocked_reasons: list of failed gate names
        - gates: detailed gate results
    """
    gates_result = evaluate_promotion_gates(
        candidate_recall=candidate_metrics.get("recall", 0.0),
        candidate_ap=candidate_metrics.get("ap", 0.0),
        candidate_orange_rate=candidate_metrics.get("orange_rate", 0.0),
        candidate_latency_ms=candidate_metrics.get("latency_ms", 0.0),
        prod_ap=prod_metrics.get("ap") if prod_metrics else None,
        gates_config=gates_config,
        candidate_quality_metrics=candidate_quality_metrics,
        prod_quality_metrics=prod_quality_metrics,
    )

    blocked_reasons = [
        gate_name
        for gate_name, result in gates_result["gates"].items()
        if not result["passed"]
    ]

    return {
        "passed": gates_result["all_passed"],
        "blocked": not gates_result["all_passed"],
        "blocked_reasons": blocked_reasons,
        "gates": gates_result["gates"],
    }


def transition_model_stage(
    registered_model_name: str,
    version: str,
    target_stage: str,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Transition a model version to a new stage in MLflow.

    Args:
        registered_model_name: Name of the registered model
        version: Version number
        target_stage: Target stage (test, prod, archived)
        tracking_uri: MLflow tracking URI (optional)

    Returns:
        Dict with:
        - success: bool
        - registered_model_name: str
        - version: str
        - new_stage: str
        - previous_stage: str (if known)
    """
    try:
        import mlflow
    except ImportError:
        raise ImportError("MLflow is required for transition_model_stage")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

    try:
        # Aliases replaced MLflow stages (deprecated since 2.9). A "stage" here is
        # an alias of the same name pointing at exactly one version. Capture the
        # alias the version held before re-aliasing, for reporting parity.
        try:
            previous_aliases = list(
                client.get_model_version(registered_model_name, str(version)).aliases
            )
            previous_stage = previous_aliases[0] if previous_aliases else None
        except Exception:
            previous_stage = None

        client.set_registered_model_alias(
            name=registered_model_name,
            alias=target_stage,
            version=str(version),
        )
        return {
            "success": True,
            "registered_model_name": registered_model_name,
            "version": str(version),
            "new_stage": target_stage,
            "previous_stage": previous_stage,
        }
    except Exception as e:
        return {
            "success": False,
            "registered_model_name": registered_model_name,
            "version": str(version),
            "target_stage": target_stage,
            "error": str(e),
        }


def resolve_model_artifacts(
    registered_model_name: str,
    stage: str = "prod",
    tracking_uri: str | None = None,
) -> dict[str, str]:
    """Resolve model artifacts (S3 URIs from MinIO) for a given stage.

    Args:
        registered_model_name: Name of the registered model
        stage: Model stage (prod, test, candidate)
        tracking_uri: MLflow tracking URI (optional)

    Returns:
        Dict with:
        - artifact_uri: str, S3 URI pointing to model artifacts in MinIO
        - stage: str
        - registered_model_name: str
    """
    try:
        import mlflow
    except ImportError:
        raise ImportError("MLflow is required for resolve_model_artifacts")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

    try:
        # Stages are deprecated; resolve via the alias of the same name.
        try:
            model_version = client.get_model_version_by_alias(
                registered_model_name, stage
            )
        except Exception:
            raise ValueError(f"No model version found for stage {stage}")

        artifact_uri = model_version.source

        return {
            "artifact_uri": artifact_uri,
            "stage": stage,
            "registered_model_name": registered_model_name,
            "version": str(model_version.version),
        }
    except Exception as e:
        raise ValueError(
            f"Failed to resolve artifacts for {registered_model_name} in stage {stage}: {e}"
        )


def promote_model_with_gates(
    registered_model_name: str,
    version: str,
    target_stage: str,
    candidate_metrics: dict[str, float],
    prod_metrics: dict[str, float] | None = None,
    gates_config_path: str | Path | None = None,
    tracking_uri: str | None = None,
    candidate_quality_metrics: dict[str, Any] | None = None,
    prod_quality_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Production workflow: evaluate gates, transition MLflow, resolve artifacts.

    This is the main entry point for promotion. It:
    1. Loads gates config (from file or uses provided)
    2. Evaluates gates on candidate metrics
    3. If gates pass: transitions model in MLflow
    4. If gates pass: resolves artifacts from MinIO

    Args:
        registered_model_name: Name of the registered model
        version: Model version to promote
        target_stage: Target stage (test, prod, archived)
        candidate_metrics: Candidate model metrics
        prod_metrics: Production model metrics (optional)
        gates_config_path: Path to promotion_gates.yaml (default: configs/promotion_gates.yaml)
        tracking_uri: MLflow tracking URI (optional)

    Returns:
        Dict with:
        - success: bool, True if promotion succeeded (gates passed + transition succeeded)
        - gates_passed: bool, True if all gates passed
        - blocked_reasons: list of gate names that failed (if gates failed)
        - transition: transition result (if gates passed)
        - artifacts: artifact resolution result (if transition succeeded)
    """
    if gates_config_path is None:
        gates_config_path = Path("configs/promotion_gates.yaml")
    else:
        gates_config_path = Path(gates_config_path)

    # Load gates config from file. Read the full content before parsing so the
    # loader does not depend on stream chunking semantics (a file handle whose
    # read() never signals EOF would make yaml.safe_load(f) loop forever).
    with open(gates_config_path, "r") as f:
        gates_config = yaml.safe_load(f.read())

    # Evaluate gates
    gate_decision = evaluate_gates_for_promotion(
        registered_model_name=registered_model_name,
        candidate_metrics=candidate_metrics,
        gates_config=gates_config,
        prod_metrics=prod_metrics,
        candidate_quality_metrics=candidate_quality_metrics,
        prod_quality_metrics=prod_quality_metrics,
    )

    # If gates fail, return early with decision
    if not gate_decision["passed"]:
        return {
            "success": False,
            "gates_passed": False,
            "blocked_reasons": gate_decision["blocked_reasons"],
            "gates": gate_decision["gates"],
        }

    # Gates passed, attempt transition
    transition_result = transition_model_stage(
        registered_model_name=registered_model_name,
        version=version,
        target_stage=target_stage,
        tracking_uri=tracking_uri,
    )

    if not transition_result["success"]:
        return {
            "success": False,
            "gates_passed": True,
            "transition": transition_result,
        }

    # Transition succeeded, resolve artifacts
    try:
        artifacts = resolve_model_artifacts(
            registered_model_name=registered_model_name,
            stage=target_stage,
            tracking_uri=tracking_uri,
        )
    except ValueError as e:
        return {
            "success": False,
            "gates_passed": True,
            "transition": transition_result,
            "artifact_error": str(e),
        }

    return {
        "success": True,
        "gates_passed": True,
        "transition": transition_result,
        "artifacts": artifacts,
    }


__all__ = [
    "evaluate_gates_for_promotion",
    "transition_model_stage",
    "resolve_model_artifacts",
    "promote_model_with_gates",
]
