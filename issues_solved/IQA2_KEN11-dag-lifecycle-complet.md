# IQA2_KEN11 — Finaliser DAG IQA_lifecycle (AIR)

**Type :** AFK · **Charge :** — · **Dates :** 17/06 → 18/06

## What to build

Finaliser le DAG `iqa_lifecycle` en remplaçant les placeholders par les vraies étapes : `dataset → train → eval → gates → mlflow → promotion → reload`.

## Acceptance criteria

- [ ] Tâche dataset (candidate_builder)
- [ ] Tâche train (train_feature_ae versionné)
- [ ] Tâche eval (evaluate_feature_ae)
- [ ] Tâche gates (bloquantes)
- [ ] Tâche mlflow (logging + registered model)
- [ ] Tâche promotion + tâche reload (model_loader)
- [ ] DAG exécutable de bout en bout (run de démo)

## Blocked by

- IQA1_KEN09 (DAG skeleton)
- IQA2_KEN02, KEN03, KEN04, KEN06, KEN09, KEN14
