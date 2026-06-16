# IQA2_KEN03 — train_feature_ae v2/v3 candidats versionnés (MOD)

**Type :** AFK · **Charge :** — · **Dates :** 14/06 → 15/06

## What to build

Implémenter `train_feature_ae` produisant des candidats versionnés `v2` et `v3` (cf. `scripts/train_feature_ae.py`), reproductibles et traçables (dataset_version, git commit).

## Acceptance criteria

- [ ] Entraînement produit un candidat versionné (`v2`, `v3`)
- [ ] Version du modèle liée à la `dataset_version` et au git commit
- [ ] Checkpoint stocké dans MinIO + manifest
- [ ] Reproductibilité (seed / config figée)
- [ ] Smoke test d'entraînement court

## Blocked by

- IQA2_KEN01 (interface Feature AE)
- IQA2_KEN02 (candidate_builder good only)
