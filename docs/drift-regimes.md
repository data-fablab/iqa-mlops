# Regimes de drift IQA

Ce document resume les deux regimes utiles a la demo IQA actuelle. Il remplace
les anciens regimes generiques par les scenarios reellement provisionnes dans le
repo.

## Regime 1 - Lifecycle Piece B

Scenario technique :

```text
production_replay_natural_piece_b_full
```

Objectif : montrer la mise en production controlee sur Piece B. Le modele part
d'un bootstrap faible, apprend sur plusieurs cycles courts, puis les gates
promeuvent uniquement les checkpoints utiles.

Preuves attendues :

- cycles lifecycle progressifs ;
- taille de train set croissante ;
- epochs visibles par cycle ;
- promotions localisation/classification separees ;
- registry final dans MLflow.

Dashboard principal :

```text
IQA - Lifecycle MLOps
```

## Regime 2 - Drift P4 et correction ciblee

Scenario technique :

```text
production_replay_natural_piece_b_to_piece_a_p4_drift
```

Objectif : montrer le regime etabli. Piece B sert de reference, P4 arrive
progressivement, la nouveaute ROI depasse le seuil, puis Airflow declenche une
correction ciblee.

Le drift n'est pas decide par le libelle du replay. La preuve principale est la
rupture ROI :

- `iqa_drift_roi_mask_novelty_rate`
- `iqa_drift_roi_mask_nn_distance`
- `iqa_drift_context_events_total`
- `iqa_drift_status`
- `iqa_drift_trigger_lifecycle`

Le DAG `iqa_drift_piece_a_p4` produit un `drift_context.json`. Le DAG dedie
`iqa_drift_correction_lifecycle` consomme ce contexte, entraine quelques epochs
sur un train set cible et publie les metriques lifecycle habituelles :

- `iqa_lifecycle_train_set_size`
- `iqa_lifecycle_epoch_current`
- `iqa_lifecycle_phase_active`
- `iqa_lifecycle_gate_value`
- `iqa_lifecycle_promotion_total`
- `iqa_lifecycle_promotion_selected_epoch`
- `iqa_lifecycle_final_model_info`

Dashboard principal :

```text
IQA - Drift P4
```

## Pourquoi ne pas utiliser `domain_ratio` comme preuve principale ?

`domain_ratio` decrit la composition du replay. Il est utile en trace, mais il ne
prouve pas que la piece observee sort du referentiel metier. La distance entre
masques ROI Piece B et P4 est plus credible pour la demo : elle mesure une
rupture geometrique observable avant la correction.

## Commandes Airflow

Depuis `deploy/` :

```bash
docker compose --env-file ../.env exec -T airflow-webserver airflow dags unpause iqa_drift_piece_a_p4
docker compose --env-file ../.env exec -T airflow-webserver airflow dags unpause iqa_drift_correction_lifecycle
docker compose --env-file ../.env exec -T airflow-webserver airflow dags trigger iqa_drift_piece_a_p4
docker compose --env-file ../.env exec -T airflow-webserver airflow dags list-runs -d iqa_drift_correction_lifecycle
```

## Garde-fous

- La CI ne declenche jamais d'entrainement.
- MLflow Registry reste la source de verite des promotions.
- MinIO stocke les checkpoints et artefacts lourds.
- Les dashboards ne doivent pas exposer de chemins locaux, masques ou images
  industrielles.
