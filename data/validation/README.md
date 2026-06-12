# Validation Set Contract

`validation_set_v001` is frozen before replay. It is excluded from bootstrap,
calibration, replay plans, and candidate Feature-AE datasets.

Required invariant:

```text
bootstrap_events ∩ replay_events ∩ validation_set_v001 = empty
```

The validation set must be stored as manifests and referenced by deterministic
IDs; heavy images stay outside Git.
