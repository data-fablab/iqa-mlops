# IQA3_NAT16 — Durable lifecycle triggers

## Objectif

Brancher le déclenchement Airflow du lifecycle Feature-AE sur les signaux réels et persistants du metadata store PostgreSQL.

## Signaux consommés

### Replay naturel

Le scénario `production_replay_natural` compte uniquement les prédictions disposant d’un feedback :

- source `oracle_gt` ;
- fermé ;
- verdict `conforme` ;
- `eligible_for_train=true`.

Le lifecycle est déclenché à partir de 50 nouveaux feedbacks conformes et éligibles.

### Drift versionné

Le scénario `drift_domain_extension` lit le dernier événement persistant de `scenario_version_events`.

Le lifecycle est déclenché lorsque :

- `drift_confirmed=true`, ou
- `lifecycle_status=drift_confirmed`.

### ROI

Le taux `roi_fail_rate` est calculé sur une fenêtre configurable de prédictions récentes et conservé dans le signal pour audit. Il ne constitue pas un déclencheur métier dans le contrat actuel.

## Idempotence

Chaque décision est enregistrée dans `lifecycle_trigger_events`.

Les événements déclenchants contiennent :

- les `prediction_id` consommés ;
- les identifiants de drift consommés ;
- le watermark du dernier feedback ;
- le taux ROI observé ;
- le dataset candidat ;
- la raison de la décision.

Après redémarrage du repository ou nouveau polling Airflow, les mêmes prédictions et événements drift ne peuvent pas déclencher une seconde fois le lifecycle.

Le DAG `iqa_lifecycle_trigger` utilise également `max_active_runs=1` pour empêcher le chevauchement de deux pollings horaires.

## Architecture Airflow

Le DAG conserve l’architecture conteneurisée du projet :

- `DockerOperator` pour collecter et persister les signaux ;
- `ShortCircuitOperator` pour le chemin sans déclenchement ;
- `TriggerDagRunOperator` pour lancer `iqa_lifecycle`.

Deux branches indépendantes sont exécutées :

- naturel ;
- drift.

Le conteneur produit un JSON lisible dans les logs et une dernière ligne JSON compacte destinée à XCom.

## Validations

### Suite complète

Fichier :

`full_ci_20260626T225510Z.log`

Résultat :

- Ruff : OK
- Pytest : 701 passés
- 11 ignorés
- 0 échec

### PostgreSQL réel

Fichier :

`postgres_contract_20260626T230102Z.log`

Résultat :

- scénario naturel `49 → aucun trigger` ;
- ajout du cinquantième feedback → trigger ;
- recréation du repository → aucun redéclenchement ;
- drift non confirmé → aucun trigger ;
- drift confirmé → trigger ;
- recréation du repository → aucun redéclenchement ;
- 2 tests passés ;
- statut 0.

### Runtime Airflow réel

Fichier :

`airflow_dag_runtime_20260626T230207Z.log`

Résultat :

- import par le vrai `DagBag` Airflow ;
- aucune erreur d’import ;
- planification `@hourly` ;
- `max_active_runs=1` ;
- six tâches naturel et drift présentes ;
- statut 0.
