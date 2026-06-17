# 10 - Lifecycle (3/4) : gates + enregistrement MLflow en conteneurs

Type : AFK

## What to build

Reecrire `gates` (verification des seuils de promotion) et `mlflow`
(enregistrement du modele dans le MLflow Registry) en taches conteneur. Respecter
l'isolation des modeles par scenario (ADR 0006) :
`feature_ae__production_replay_natural` vs `feature_ae__drift_domain_extension`.

## Acceptance criteria

- [ ] `gates` evalue les `configs/promotion_gates.yaml` en conteneur et bloque si echec
- [ ] `mlflow` enregistre le modele dans le Registry avec le nom isole par scenario
- [ ] Echec de gate stoppe le DAG avant enregistrement
- [ ] Import DagBag vert

## Blocked by

- 09 - Lifecycle (2/4) : train + eval
