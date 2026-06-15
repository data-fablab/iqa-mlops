# IQA2_KEN14 — Brancher model_loader (INF)

**Type :** AFK · **Charge :** — · **Dates :** 18/06 → 19/06

## What to build

Brancher le `model_loader` (`iqa.inference`) sur :

- MLflow `prod` par `scenario_id`
- artefact résolu depuis MinIO

Permet à l'inférence de charger le modèle prod courant et de se recharger après promotion.

## Acceptance criteria

- [ ] Chargement du modèle `prod` du bon `scenario_id` via MLflow
- [ ] Artefact (checkpoint) résolu depuis MinIO
- [ ] Reload après promotion (cohérent avec le DAG `reload`)
- [ ] Test de chargement prod par `scenario_id`

## Blocked by

- IQA2_KEN07 (registered models par scenario_id)
- IQA2_KEN09 (promotion)
