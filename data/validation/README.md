# Validation And Calibration Contracts

`validation_set_replay_representative_v001` is the frozen representative gate
set for the Feature-AE MVP lifecycle. It is excluded from bootstrap,
calibration, replay plans, candidate Feature-AE datasets and threshold
calibration.

`calibration_good_reference_v001` is a good-only set reserved for Feature-AE
threshold calibration. It is excluded from bootstrap, replay, train and
validation.

Required invariant:

```text
bootstrap ∩ calibration ∩ replay ∩ validation = empty
```

The Casting dataset is a historical source replayed through IQA ingestion
contracts. Runtime production images will be stored in MinIO and traced in
PostgreSQL, but the MVP remains locally testable through deterministic
manifests. Heavy images stay outside Git and are versioned through DVC/MinIO.
