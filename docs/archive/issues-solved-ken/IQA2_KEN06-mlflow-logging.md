# IQA2_KEN06 — Logger runs MLflow complets (MLF)

**Type :** AFK · **Charge :** — · **Dates :** 16/06 → 16/06

## What to build

Logger les runs MLflow avec la traçabilité complète : `params`, `metrics`, `artifacts`, `git commit`, `dataset_version`, `scenario_id`.

## Acceptance criteria

- [ ] `params` loggés
- [ ] `metrics` loggées (AP, recall, Orange rate, latency)
- [ ] `artifacts` loggés (checkpoint/manifest, rapport eval)
- [ ] Tags `git commit`, `dataset_version`, `scenario_id` présents
- [ ] Test vérifiant la présence des champs de traçabilité

## Blocked by

- IQA1_KEN07 (MLflow source de vérité)
- IQA2_KEN04 (evaluate)
