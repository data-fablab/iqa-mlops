from pathlib import Path

from iqa.monitoring.lifecycle import (
    FEATURE_AE_V002_DATASET_VERSION,
    FEATURE_AE_V003_DATASET_VERSION,
    LifecycleSignal,
    evaluate_lifecycle_signal,
    should_trigger_lifecycle,
)


def test_natural_replay_waits_for_50_oracle_conformes() -> None:
    signal = LifecycleSignal(
        scenario_id="production_replay_natural",
        conforming_validated_count=49,
        drift_confirmed=False,
    )

    decision = evaluate_lifecycle_signal(signal)

    assert not decision.trigger_lifecycle
    assert decision.trigger_reason == "natural_waiting_for_50_oracle_conformes"
    assert decision.candidate_dataset_version is None


def test_natural_replay_triggers_feature_ae_v002_at_50_oracle_conformes() -> None:
    signal = LifecycleSignal(
        scenario_id="production_replay_natural",
        conforming_validated_count=50,
        drift_confirmed=False,
    )

    decision = evaluate_lifecycle_signal(signal)

    assert decision.trigger_lifecycle
    assert decision.trigger_reason == "natural_50_oracle_conformes"
    assert decision.candidate_dataset_version == FEATURE_AE_V002_DATASET_VERSION
    assert should_trigger_lifecycle(signal)


def test_natural_train_replay_triggers_feature_ae_at_50_oracle_conformes() -> None:
    signal = LifecycleSignal(
        scenario_id="production_replay_natural_train_v004",
        conforming_validated_count=50,
        drift_confirmed=False,
    )

    decision = evaluate_lifecycle_signal(signal)

    assert decision.trigger_lifecycle
    assert decision.trigger_reason == "natural_50_oracle_conformes"
    assert decision.candidate_dataset_version == FEATURE_AE_V002_DATASET_VERSION
    assert should_trigger_lifecycle(signal)


def test_drift_replay_waits_for_confirmed_drift() -> None:
    signal = LifecycleSignal(
        scenario_id="drift_domain_extension",
        conforming_validated_count=50,
        drift_confirmed=False,
    )

    decision = evaluate_lifecycle_signal(signal)

    assert not decision.trigger_lifecycle
    assert decision.trigger_reason == "drift_not_confirmed"
    assert decision.candidate_dataset_version is None


def test_drift_replay_triggers_feature_ae_v003_on_confirmed_drift() -> None:
    signal = LifecycleSignal(
        scenario_id="drift_domain_extension",
        conforming_validated_count=0,
        drift_confirmed=True,
    )

    decision = evaluate_lifecycle_signal(signal)

    assert decision.trigger_lifecycle
    assert decision.trigger_reason == "drift_confirmed"
    assert decision.candidate_dataset_version == FEATURE_AE_V003_DATASET_VERSION
    assert should_trigger_lifecycle(signal)


def test_piece_a_p4_drift_triggers_lifecycle_on_confirmed_drift() -> None:
    signal = LifecycleSignal(
        scenario_id="production_replay_natural_piece_b_to_piece_a_p4_drift",
        conforming_validated_count=0,
        drift_confirmed=True,
    )

    decision = evaluate_lifecycle_signal(signal)

    assert decision.trigger_lifecycle
    assert decision.trigger_reason == "drift_piece_a_p4_confirmed"
    assert decision.candidate_dataset_version == FEATURE_AE_V003_DATASET_VERSION


def test_unknown_scenario_does_not_trigger_lifecycle() -> None:
    signal = LifecycleSignal(
        scenario_id="manual_experiment",
        conforming_validated_count=100,
        drift_confirmed=True,
    )

    decision = evaluate_lifecycle_signal(signal)

    assert not decision.trigger_lifecycle
    assert decision.trigger_reason == "unsupported_scenario"


def test_lifecycle_docs_record_data_event_triggers() -> None:
    docs = Path("docs/model-lifecycle.md").read_text(encoding="utf-8")
    normalized_docs = " ".join(docs.split())

    assert "50" in normalized_docs
    assert "oracle_gt" in normalized_docs
    assert "drift_confirmed=true" in normalized_docs
    assert "La CI ne declenche jamais" in normalized_docs
