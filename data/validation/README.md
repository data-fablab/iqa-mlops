# Validation And Calibration Contracts

`validation_set_replay_gate_v001` is the frozen AUPIMO promotion gate panel
for short progressive Feature-AE lifecycle runs. It preserves the historical
image-level reference panel where the bootstrap model has a non-zero AUPIMO
signal, so the fast gate can compare candidates without collapsing the
localisation metric.

`validation_set_replay_gate_v002` is the default fast candidate promotion gate
panel. It keeps the defect reserve available in validation/gate data and adds
120 good pieces held out from bootstrap, calibration and natural replay. It has
defect mask coverage through `validation_gt_masks_v001.csv`, so AUPIMO remains
defined during progressive lifecycle runs.

`validation_set_replay_gate_v003` is the Scenario-B fixed holdout panel split
out of the natural replay source before progressive training starts: 10 defect
pieces and 120 good pieces are reserved while
`casting_flux_replay_plan_natural_train_v004.csv` carries the remaining replay
stream. The replay defect pieces are not covered by `validation_gt_masks_v001`,
so v003 is useful for replay split auditing but must not be used as the AUPIMO
promotion gate.

`validation_set_replay_representative_v001` is the larger representative audit
set for full validation and release evidence. It is excluded from bootstrap,
calibration, replay plans, candidate Feature-AE datasets and threshold
calibration, but it is not the default fast promotion gate.

`calibration_good_reference_v001` is a good-only set reserved for Feature-AE
threshold calibration. It is excluded from bootstrap, replay, train and
validation.

`scripts/build_validation_gate_v2.py` rebuilds
`validation_set_replay_gate_v002.csv` deterministically from the source
metadata. The historical source contains no defect pieces left outside all
existing replay/validation reserves, so v002 uses reserved validation defects
and fresh held-out good pieces. Defect coverage of late replay events remains
limited until new labeled defect reserves are collected.

`scripts/build_replay_holdout_split.py` rebuilds Scenario B deterministically:
`data/metadata/casting_flux_replay_plan_natural_train_v004.csv` plus
`data/validation/validation_set_replay_gate_v003.csv`. The train replay and
gate holdout are disjoint and their union reconstructs the natural replay v003
source events.

Required invariant:

```text
bootstrap ∩ calibration ∩ replay ∩ validation = empty
```

The Casting dataset is a historical source replayed through IQA ingestion
contracts. Runtime production images will be stored in MinIO and traced in
PostgreSQL, but the MVP remains locally testable through deterministic
manifests. Heavy images stay outside Git and are versioned through DVC/MinIO.
