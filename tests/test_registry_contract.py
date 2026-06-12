from __future__ import annotations

from iqa.registry import ModelRegistryRef, registered_model_name


def test_registered_model_name_is_isolated_by_scenario() -> None:
    assert registered_model_name("production_replay_natural") == "feature_ae__production_replay_natural"
    assert registered_model_name("drift_domain_extension") == "feature_ae__drift_domain_extension"


def test_registry_ref_declares_mlflow_as_source_of_truth() -> None:
    ref = ModelRegistryRef(
        scenario_id="production_replay_natural",
        registered_model_name="feature_ae__production_replay_natural",
    )

    assert ref.to_dict()["stage"] == "prod"
    assert ref.to_dict()["source_of_truth"] == "mlflow_registry"
