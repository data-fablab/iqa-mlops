# IQA3_NAT17 — MLflow production model runtime reload

## Objective

Propagate a model promoted through the MLflow Model Registry to the running `iqa-inference` service without restarting the container.

The served Feature-AE bundle contains:

- the model checkpoint;
- the model manifest;
- calibrated decision thresholds;
- the Feature-AE scoring contract;
- the checkpoint SHA256;
- the MLflow LoggedModel and Registry identifiers.

## Implemented flow

1. A promoted lifecycle candidate is packaged as an MLflow 3 LoggedModel.
2. The LoggedModel is registered under the scenario-specific registered model.
3. The `prod` alias is assigned to the new Registry version.
4. The API delegates the protected reload request to `iqa-inference`.
5. `iqa-inference` downloads and validates the complete bundle.
6. The checkpoint, manifest, thresholds and scoring contract are loaded before exposure.
7. The active runtime is replaced atomically.
8. A failed reload leaves the previous runtime active.

## Live Registry promotion

- Scenario: `production_replay_natural`
- Registered model: `feature_ae__production_replay_natural`
- Alias: `prod`
- Registry version: `10`
- MLflow model ID: `m-414e88fca7c34db9ab98ec61645f725d`
- Model URI: `models:/m-414e88fca7c34db9ab98ec61645f725d`
- Promoted Feature-AE version: `rd_feature_ae_gated_natural_cycle_004`

## Real runtime reload

The reload was executed through the protected Kong route:

`POST /api/admin/reload-model`

Result:

- HTTP status: `200`
- Reload status: `reloaded`
- Previous model: `rd_feature_ae_gated_v001_bootstrap`
- Active model: `rd_feature_ae_gated_natural_cycle_004`
- Active Registry version: `10`

## Real before/after inference proof

The same real defective image recovered from the DVC dataset was used before and after the runtime reload.

Source:

`Casting_class1/test/defective/2022-02-23_13_15_19_383-2_1_2.jpg`

Before reload:

- Feature-AE version: `rd_feature_ae_gated_v001_bootstrap`
- Anomaly score: `0.7933963537216187`
- Decision: `Vert`

After reload:

- Feature-AE version: `rd_feature_ae_gated_natural_cycle_004`
- Anomaly score: `1.187240481376648`
- Decision: `Rouge`

The changed model version, score and decision demonstrate that the promoted MLflow model became the actual model used by the running inference service.

The raw dataset image is not committed. Its source path, size, MD5 and SHA256 are recorded in `real_input_metadata.json`.

## Final validation

Validation was executed in an isolated environment to avoid contamination from live `.env` variables.

- Ruff: passed
- Pytest: `700 passed`
- Skipped: `20`
- Warnings: `5` known MLflow warnings
- Runtime reload tests: passed
- MLflow 3 bundle and Registry tests: passed
- API security and delegation tests: passed

## Main evidence

- `mlflow_prod_registration.json`
- `promoted_serving_manifest.json`
- `reload_prod_response.json`
- `model_before_reload.json`
- `prediction_before_reload.json`
- `prediction_after_reload.json`
- `prediction_before_after_comparison.json`
- `real_input_metadata.json`
- `full_ci_final_isolated.txt`
