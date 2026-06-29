from __future__ import annotations

from typing import Callable

from scripts import build_piece_a_p4_drift_trigger_conf as builder
from scripts import run_monitoring


def _conf_for_phase(phase: builder.Phase) -> dict:
    rows = builder.read_plan(builder.DEFAULT_PLAN)
    return builder.build_conf(
        phase=phase,
        rows=rows,
        epochs=16,
        max_events=None,
        lifecycle_interval=50,
        ml_image="iqa-ml:local",
        image_root="/opt/iqa/iqa-mlops/.cache/iqa/source_datasets/hss-iad",
    )


def _monitoring_args_from_conf(conf: dict) -> list[str]:
    return [
        "iqa-run-monitoring",
        "--scenario-id",
        str(conf["scenario_id"]),
        "--conforming-validated-count",
        str(conf["conforming_validated_count"]),
        "--drift-confirmed",
        str(conf["drift_confirmed"]),
        "--roi-fail-rate",
        str(conf["roi_fail_rate"]),
        "--source-domain",
        str(conf["source_domain"]),
        "--window-events",
        str(conf["window_events"]),
        "--domain-ratio",
        str(conf["domain_ratio"]),
        "--alert-rate",
        str(conf["alert_rate"]),
        "--red-rate",
        str(conf["red_rate"]),
        "--unexpected-red-rate",
        str(conf["unexpected_red_rate"]),
        "--oracle-fn-rate",
        str(conf["oracle_fn_rate"]),
        "--critical-window-count",
        str(conf["critical_window_count"]),
        "--api-url",
        "",
        "--thresholds-config",
        str(conf["thresholds_config"]),
    ]


def test_piece_a_p4_clear_window_does_not_trigger_lifecycle(
    run_boundary_script: Callable[[object, list[str]], dict],
) -> None:
    conf = _conf_for_phase("clear")

    result = run_boundary_script(run_monitoring, _monitoring_args_from_conf(conf))

    assert conf["scenario_validation"]["expected_monitoring_status"] == "clear"
    assert result["drift_evaluation"]["status"] == "clear"
    assert result["trigger_lifecycle"] is False


def test_piece_a_p4_suspected_window_waits_for_confirmation(
    run_boundary_script: Callable[[object, list[str]], dict],
) -> None:
    conf = _conf_for_phase("suspected")

    result = run_boundary_script(run_monitoring, _monitoring_args_from_conf(conf))

    assert conf["critical_window_count"] == 0
    assert result["drift_evaluation"]["status"] == "suspected"
    assert result["drift_suspected"] is True
    assert result["drift_confirmed"] is False
    assert result["trigger_lifecycle"] is False


def test_piece_a_p4_confirmed_window_triggers_correction(
    run_boundary_script: Callable[[object, list[str]], dict],
) -> None:
    conf = _conf_for_phase("confirmed")

    result = run_boundary_script(run_monitoring, _monitoring_args_from_conf(conf))

    assert conf["drift_confirmed"] is False
    assert conf["critical_window_count"] == 1
    assert result["drift_evaluation"]["status"] == "confirmed"
    assert result["drift_confirmed"] is True
    assert result["trigger_lifecycle"] is True
    assert result["trigger_reason"] == "drift_piece_a_p4_confirmed"


def test_piece_a_p4_trigger_conf_uses_stable_piece_b_models_for_correction() -> None:
    conf = _conf_for_phase("confirmed")

    assert conf["mode"] == "progressive-train"
    assert conf["candidate_init_policy"] == "active"
    assert conf["initial_classification_registered_model"] == builder.STABLE_CLASSIFICATION_MODEL
    assert conf["initial_localization_registered_model"] == builder.STABLE_LOCALIZATION_MODEL
    assert conf["require_mlflow_registry"] is True
    assert conf["reference_eval_manifest"] == "data/validation/validation_set_piece_b_to_piece_a_p4_drift_v001.csv"
    assert conf["classification_selection_manifest"] == "data/validation/classification_selection_piece_b_to_piece_a_p4_drift_v001.csv"
    assert conf["epochs"] == 16
    assert conf["scenario_validation"]["plan_summary"]["p4_event_count"] == 93
