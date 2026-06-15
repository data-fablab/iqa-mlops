# IQA Phase 1 Data Validation

## Volumes

- piece_events: 962
- defective piece_events: 40
- images referenced by piece_events: 2183
- bootstrap events: 50
- validation_set_v001 events: 20
- calibration_set_v001 events: 60
- production_replay_natural events: 832
- drift_domain_extension events: 832

## Invariant

`bootstrap ∩ calibration ∩ replay ∩ validation = empty`

- bootstrap_vs_validation: 0
- bootstrap_vs_calibration: 0
- bootstrap_vs_replay: 0
- validation_vs_calibration: 0
- validation_vs_replay: 0
- calibration_vs_replay: 0

## Notes

- `validation_set_v001` is frozen and used for metric-best selection.
- `calibration_set_v001` is good-only and reserved for Feature-AE threshold calibration.
- Replay plans keep defects for oracle feedback but exclude bootstrap, validation and calibration events.
