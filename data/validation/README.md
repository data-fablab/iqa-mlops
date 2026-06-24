# Validation And Calibration Contracts

`validation_set_replay_gate_v001` is the frozen AUPIMO promotion gate panel
for short progressive Feature-AE lifecycle runs. It preserves the historical
image-level reference panel where the bootstrap model has a non-zero AUPIMO
signal, so the fast gate can compare candidates without collapsing the
localisation metric.

`validation_set_replay_representative_v001` is the larger representative audit
set for full validation and release evidence. It is excluded from bootstrap,
calibration, replay plans, candidate Feature-AE datasets and threshold
calibration, but it is not the default fast promotion gate.

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
