# IQA2_KEN01 — Interface Feature AE candidat (MOD)

**Type :** AFK · **Charge :** — · **Dates :** 13/06 → 14/06

## What to build

Finaliser l'interface du modèle Feature AE candidat (`iqa.models.feature_ae` / `iqa.training.feature_ae`) exposant : `train`, `eval`, `save`, `load`, `predict`.

## Acceptance criteria

- [ ] `train` entraîne le Feature AE sur un dataset candidat
- [ ] `eval` calcule les métriques sur un set d'évaluation
- [ ] `save` / `load` round-trip (checkpoint MinIO référencé par manifest)
- [ ] `predict` conforme au contrat modèle
- [ ] Test de round-trip save/load

## Blocked by

- IQA1_KEN03 (wrappers)
