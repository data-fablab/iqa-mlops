"""TDD tests for the multi-signal retrain policy evaluator (Issues 15-18)."""

from iqa.monitoring.retrain_policy import (
    DEFAULT_METRIC_FLOOR,
    RetrainPolicyDecision,
    RetrainPolicySignal,
    evaluate_retrain_policy,
    retrain_policy_parity_with_lifecycle,
)


# ── Issue 15: accumulation trigger (parity with evaluate_lifecycle_signal) ──


class TestAccumulationTrigger:
    def test_below_threshold_does_not_trigger(self) -> None:
        signal = RetrainPolicySignal(conforming_validated_count=49)
        decision = evaluate_retrain_policy(signal)
        assert not decision.trigger
        assert decision.primary_reason == "no_trigger"

    def test_at_threshold_triggers(self) -> None:
        signal = RetrainPolicySignal(conforming_validated_count=50)
        decision = evaluate_retrain_policy(signal)
        assert decision.trigger
        assert "accumulation" in decision.trigger_reasons
        assert decision.primary_reason == "accumulation"
        assert decision.retrain_scope == "bootstrap"

    def test_above_threshold_triggers(self) -> None:
        signal = RetrainPolicySignal(conforming_validated_count=100)
        decision = evaluate_retrain_policy(signal)
        assert decision.trigger
        assert "accumulation" in decision.trigger_reasons

    def test_custom_threshold(self) -> None:
        signal = RetrainPolicySignal(conforming_validated_count=10)
        decision = evaluate_retrain_policy(signal, min_conforming=10)
        assert decision.trigger
        assert "accumulation" in decision.trigger_reasons

    def test_parity_with_legacy_below_threshold(self) -> None:
        signal = RetrainPolicySignal(conforming_validated_count=30, drift_confirmed=False)
        assert retrain_policy_parity_with_lifecycle(signal)

    def test_parity_with_legacy_above_threshold(self) -> None:
        signal = RetrainPolicySignal(conforming_validated_count=50, drift_confirmed=False)
        assert retrain_policy_parity_with_lifecycle(signal)

    def test_decision_shape(self) -> None:
        signal = RetrainPolicySignal(conforming_validated_count=50)
        decision = evaluate_retrain_policy(signal)
        assert isinstance(decision, RetrainPolicyDecision)
        assert isinstance(decision.trigger_reasons, list)
        assert decision.candidate_dataset_version is not None
        d = decision.to_dict()
        assert "trigger" in d
        assert "trigger_reasons" in d
        assert "primary_reason" in d


# ── Issue 16: metric floor trigger ──


class TestMetricFloorTrigger:
    def test_prod_below_aupimo_target_triggers(self) -> None:
        signal = RetrainPolicySignal(
            prod_metrics={"pixel_aupimo_1e-5_1e-3": 0.07, "pixel_ap": 0.30},
        )
        decision = evaluate_retrain_policy(signal)
        assert decision.trigger
        assert "metric_floor" in decision.trigger_reasons
        assert decision.retrain_scope == "full_domain_good"

    def test_prod_below_pixel_ap_target_triggers(self) -> None:
        signal = RetrainPolicySignal(
            prod_metrics={"pixel_aupimo_1e-5_1e-3": 0.20, "pixel_ap": 0.10},
        )
        decision = evaluate_retrain_policy(signal)
        assert decision.trigger
        assert "metric_floor" in decision.trigger_reasons

    def test_prod_above_both_targets_does_not_trigger(self) -> None:
        signal = RetrainPolicySignal(
            prod_metrics={"pixel_aupimo_1e-5_1e-3": 0.20, "pixel_ap": 0.30},
        )
        decision = evaluate_retrain_policy(signal)
        assert "metric_floor" not in decision.all_fired_reasons

    def test_empty_prod_metrics_does_not_trigger(self) -> None:
        signal = RetrainPolicySignal(prod_metrics={})
        decision = evaluate_retrain_policy(signal)
        assert "metric_floor" not in decision.all_fired_reasons

    def test_custom_floor_targets(self) -> None:
        signal = RetrainPolicySignal(
            prod_metrics={"pixel_aupimo_1e-5_1e-3": 0.50},
        )
        decision = evaluate_retrain_policy(
            signal, floor_targets={"pixel_aupimo_1e-5_1e-3": 0.60}
        )
        assert decision.trigger
        assert "metric_floor" in decision.trigger_reasons

    def test_metric_floor_never_used_as_gate(self) -> None:
        signal = RetrainPolicySignal(
            prod_metrics={"pixel_aupimo_1e-5_1e-3": 0.07},
        )
        decision = evaluate_retrain_policy(signal)
        assert decision.trigger
        assert decision.retrain_scope == "full_domain_good"
        assert decision.candidate_dataset_version == "feature_ae_full_retrain"

    def test_aupimo_unstable_no_trigger_when_above(self) -> None:
        signal = RetrainPolicySignal(
            prod_metrics={"pixel_aupimo_1e-5_1e-3": 0.16, "pixel_ap": 0.25},
        )
        decision = evaluate_retrain_policy(signal)
        assert "metric_floor" not in decision.all_fired_reasons


# ── Issue 17: PatchCore drift trigger ──


class TestDriftTrigger:
    def test_drift_confirmed_triggers(self) -> None:
        signal = RetrainPolicySignal(
            drift_confirmed=True,
            drift_triggering_class="Casting_class2",
        )
        decision = evaluate_retrain_policy(signal)
        assert decision.trigger
        assert "drift" in decision.trigger_reasons
        assert decision.triggering_class == "Casting_class2"
        assert decision.retrain_scope == "incremental_coverage"

    def test_high_ood_ratio_triggers(self) -> None:
        signal = RetrainPolicySignal(
            drift_ood_ratio=0.7,
            drift_triggering_class="Casting_class3",
        )
        decision = evaluate_retrain_policy(signal)
        assert decision.trigger
        assert "drift" in decision.trigger_reasons
        assert decision.triggering_class == "Casting_class3"

    def test_low_ood_ratio_no_trigger(self) -> None:
        signal = RetrainPolicySignal(drift_ood_ratio=0.1)
        decision = evaluate_retrain_policy(signal)
        assert "drift" not in decision.all_fired_reasons

    def test_drift_dataset_version_includes_class(self) -> None:
        signal = RetrainPolicySignal(
            drift_confirmed=True,
            drift_triggering_class="Casting_class2",
        )
        decision = evaluate_retrain_policy(signal)
        assert decision.candidate_dataset_version == "feature_ae_drift_Casting_class2"

    def test_drift_class3_incremental(self) -> None:
        signal = RetrainPolicySignal(
            drift_confirmed=True,
            drift_triggering_class="Casting_class3",
        )
        decision = evaluate_retrain_policy(signal)
        assert decision.candidate_dataset_version == "feature_ae_drift_Casting_class3"
        assert decision.retrain_scope == "incremental_coverage"

    def test_existing_accumulation_path_still_works(self) -> None:
        signal = RetrainPolicySignal(conforming_validated_count=50)
        decision = evaluate_retrain_policy(signal)
        assert decision.trigger
        assert "accumulation" in decision.trigger_reasons

    def test_custom_ood_threshold(self) -> None:
        signal = RetrainPolicySignal(
            drift_ood_ratio=0.3,
            drift_triggering_class="Casting_class2",
        )
        decision = evaluate_retrain_policy(signal, ood_ratio_threshold=0.2)
        assert decision.trigger
        assert "drift" in decision.trigger_reasons


# ── Issue 18: anti-loop + HITL escalation + multi-trigger union ──


class TestAntiLoop:
    def test_no_retrigger_when_inputs_unchanged(self) -> None:
        last_inputs = {
            "conforming_validated_count": 60,
            "drift_triggering_class": None,
            "prod_metrics": {},
        }
        signal = RetrainPolicySignal(
            conforming_validated_count=60,
            last_trigger_inputs=last_inputs,
        )
        decision = evaluate_retrain_policy(signal)
        assert not decision.trigger
        assert decision.blocked_reason == "inputs_unchanged"

    def test_retrigger_when_count_increased(self) -> None:
        last_inputs = {
            "conforming_validated_count": 50,
            "drift_triggering_class": None,
            "prod_metrics": {},
        }
        signal = RetrainPolicySignal(
            conforming_validated_count=70,
            last_trigger_inputs=last_inputs,
        )
        decision = evaluate_retrain_policy(signal)
        assert decision.trigger

    def test_retrigger_when_new_drift_class(self) -> None:
        last_inputs = {
            "conforming_validated_count": 0,
            "drift_triggering_class": "Casting_class2",
            "prod_metrics": {},
        }
        signal = RetrainPolicySignal(
            drift_confirmed=True,
            drift_triggering_class="Casting_class3",
            last_trigger_inputs=last_inputs,
        )
        decision = evaluate_retrain_policy(signal)
        assert decision.trigger

    def test_no_retrigger_same_drift_class(self) -> None:
        last_inputs = {
            "conforming_validated_count": 0,
            "drift_triggering_class": "Casting_class2",
            "prod_metrics": {},
        }
        signal = RetrainPolicySignal(
            drift_confirmed=True,
            drift_triggering_class="Casting_class2",
            last_trigger_inputs=last_inputs,
        )
        decision = evaluate_retrain_policy(signal)
        assert not decision.trigger
        assert decision.blocked_reason == "inputs_unchanged"

    def test_blocked_when_lifecycle_run_in_flight(self) -> None:
        signal = RetrainPolicySignal(
            conforming_validated_count=60,
            active_lifecycle_run=True,
        )
        decision = evaluate_retrain_policy(signal)
        assert not decision.trigger
        assert decision.blocked_reason == "lifecycle_run_in_flight"

    def test_cooldown_blocks_trigger(self) -> None:
        signal = RetrainPolicySignal(
            conforming_validated_count=60,
            seconds_since_last_attempt=100.0,
        )
        decision = evaluate_retrain_policy(signal, cooldown_seconds=900)
        assert not decision.trigger
        assert decision.blocked_reason == "cooldown_active"

    def test_cooldown_expired_allows_trigger(self) -> None:
        signal = RetrainPolicySignal(
            conforming_validated_count=60,
            seconds_since_last_attempt=1000.0,
        )
        decision = evaluate_retrain_policy(signal, cooldown_seconds=900)
        assert decision.trigger


class TestHITLEscalation:
    def test_max_failures_blocks_and_escalates(self) -> None:
        signal = RetrainPolicySignal(
            conforming_validated_count=60,
            gate_failure_count=2,
        )
        decision = evaluate_retrain_policy(signal, max_gate_failures=2)
        assert not decision.trigger
        assert decision.hitl_escalation
        assert decision.blocked_reason == "hitl_escalation_max_failures"

    def test_below_max_failures_triggers(self) -> None:
        signal = RetrainPolicySignal(
            conforming_validated_count=60,
            gate_failure_count=1,
        )
        decision = evaluate_retrain_policy(signal, max_gate_failures=2)
        assert decision.trigger
        assert not decision.hitl_escalation

    def test_zero_failures_triggers(self) -> None:
        signal = RetrainPolicySignal(
            conforming_validated_count=60,
            gate_failure_count=0,
        )
        decision = evaluate_retrain_policy(signal)
        assert decision.trigger


class TestMultiTriggerUnion:
    def test_drift_plus_floor_single_run_drift_priority(self) -> None:
        signal = RetrainPolicySignal(
            drift_confirmed=True,
            drift_triggering_class="Casting_class2",
            prod_metrics={"pixel_aupimo_1e-5_1e-3": 0.07, "pixel_ap": 0.01},
        )
        decision = evaluate_retrain_policy(signal)
        assert decision.trigger
        assert "drift" in decision.all_fired_reasons
        assert "metric_floor" in decision.all_fired_reasons
        assert decision.primary_reason == "drift"
        assert decision.retrain_scope == "incremental_coverage"

    def test_floor_plus_accumulation_floor_priority(self) -> None:
        signal = RetrainPolicySignal(
            conforming_validated_count=60,
            prod_metrics={"pixel_aupimo_1e-5_1e-3": 0.07},
        )
        decision = evaluate_retrain_policy(signal)
        assert decision.trigger
        assert "metric_floor" in decision.all_fired_reasons
        assert "accumulation" in decision.all_fired_reasons
        assert decision.primary_reason == "metric_floor"
        assert decision.retrain_scope == "full_domain_good"

    def test_all_three_triggers_drift_wins(self) -> None:
        signal = RetrainPolicySignal(
            conforming_validated_count=60,
            drift_confirmed=True,
            drift_triggering_class="Casting_class3",
            prod_metrics={"pixel_aupimo_1e-5_1e-3": 0.05, "pixel_ap": 0.01},
        )
        decision = evaluate_retrain_policy(signal)
        assert decision.trigger
        assert len(decision.all_fired_reasons) == 3
        assert decision.primary_reason == "drift"
        assert decision.retrain_scope == "incremental_coverage"
        assert decision.triggering_class == "Casting_class3"

    def test_all_reasons_logged(self) -> None:
        signal = RetrainPolicySignal(
            conforming_validated_count=60,
            drift_confirmed=True,
            drift_triggering_class="Casting_class2",
            prod_metrics={"pixel_aupimo_1e-5_1e-3": 0.07},
        )
        decision = evaluate_retrain_policy(signal)
        assert set(decision.all_fired_reasons) == {"accumulation", "metric_floor", "drift"}

    def test_single_trigger_only_one_reason(self) -> None:
        signal = RetrainPolicySignal(conforming_validated_count=50)
        decision = evaluate_retrain_policy(signal)
        assert decision.all_fired_reasons == ["accumulation"]
        assert decision.primary_reason == "accumulation"


class TestNonRegression:
    """Verify that individual trigger paths from Issues 15/16/17 remain functional."""

    def test_accumulation_only(self) -> None:
        signal = RetrainPolicySignal(conforming_validated_count=50)
        decision = evaluate_retrain_policy(signal)
        assert decision.trigger
        assert decision.primary_reason == "accumulation"

    def test_metric_floor_only(self) -> None:
        signal = RetrainPolicySignal(
            prod_metrics={"pixel_aupimo_1e-5_1e-3": 0.07, "pixel_ap": 0.01},
        )
        decision = evaluate_retrain_policy(signal)
        assert decision.trigger
        assert decision.primary_reason == "metric_floor"

    def test_drift_only(self) -> None:
        signal = RetrainPolicySignal(
            drift_confirmed=True,
            drift_triggering_class="Casting_class2",
        )
        decision = evaluate_retrain_policy(signal)
        assert decision.trigger
        assert decision.primary_reason == "drift"

    def test_no_signal_no_trigger(self) -> None:
        signal = RetrainPolicySignal()
        decision = evaluate_retrain_policy(signal)
        assert not decision.trigger
        assert decision.primary_reason == "no_trigger"
