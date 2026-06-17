# Issues - Migration Airflow DockerOperator + microservices + automatisation

Perimetre derive des ADR [0002](../docs/adr/0002-airflow-comme-orchestrateur.md),
[0007](../docs/adr/0007-architecture-services-avec-pyproject-racine.md) et
[0008](../docs/adr/0008-taches-airflow-comme-conteneurs.md).

Tranches verticales (tracer bullets). Une seule HITL (00) ; le reste est AFK.

| # | Tranche | Type | Bloque par |
|---|---------|------|------------|
| 00 | Decision registre Docker Hub / tags / secrets CI | HITL | - |
| 01 | Extras par role dans pyproject (serving/ml/data) | AFK | - |
| 02 | Dockerfile multi-stage + image iqa-api slim (tracer) | AFK | 01 |
| 03 | Image ml (inference, trainer) | AFK | 02 |
| 04 | Image data (ingestion, replay, monitoring) | AFK | 02 |
| 05 | Factory make_container_task (docker\|k8s) | AFK | 02 |
| 06 | Compose : socket Docker, reseau, lock GPU | AFK | 05 |
| 07 | DAG ingestion en conteneur | AFK | 06, 04 |
| 08 | Lifecycle 1/4 : decision + dataset | AFK | 06, 04 |
| 09 | Lifecycle 2/4 : train + eval (pool GPU) | AFK | 08, 03 |
| 10 | Lifecycle 3/4 : gates + MLflow | AFK | 09 |
| 11 | Lifecycle 4/4 : promotion + reload | AFK | 10 |
| 12 | DAG replay en conteneur | AFK | 06, 04 |
| 13 | DAG monitoring en conteneur | AFK | 06, 04 |
| 14 | CI build + push images Docker Hub (matrix) | AFK | 00, 03, 04 |
| 15 | Compose + DAGs referencent les images du registre par tag | AFK | 14 |
| 16 | Sensor de declenchement evenementiel du lifecycle | AFK | 11 |
| 17 | Overlays Compose dev / prod | AFK | 02 |
| 18 | Persistance runtime de l'ingestion (events PG / images MinIO) | AFK | 07 |
| 19 | Persistance runtime du dataset candidat (materialisation MinIO / PG) | AFK | 08 |
| 20 | Runtime train/eval : entrainement reel + checkpoint/metriques MinIO | AFK | 09, 19 |
| 21 | Runtime MLflow : enregistrement reel du run au Registry | AFK | 10, 20 |
| 22 | Runtime promotion + reload : transition Registry reelle + reload HTTP inference | AFK | 11, 21 |
| 23 | Runtime monitoring : export des metriques vers Prometheus / Grafana | AFK | 13 |

## Chemin critique

```text
01 -> 02 -> 05 -> 06 -> 08 -> 09 -> 10 -> 11 -> 16
```

Les images (03/04) et la CI (14 -> 15) se parallelisent ; 00 (HITL) doit etre tranchee avant 14.

## Lots de travail

- Microservices/images : 01, 02, 03, 04, 17
- Orchestration conteneurisee : 05, 06, 07, 08, 09, 10, 11, 12, 13
- Automatisation / registre : 00, 14, 15, 16
- Persistance runtime (data plane) : 18, 19, 20, 21, 22, 23

## Cadrage : conteneurisation DAG vs persistance runtime

Les scripts `iqa-run-*` (ex. `scripts/run_ingestion.py`) sont aujourd'hui des
**frontières "validated-summary"** : ils lisent une entrée et impriment un résumé
JSON, sans écrire dans PostgreSQL/MinIO/MLflow. Deux travaux distincts sont donc en
jeu et **ne doivent pas être confondus** :

1. **Conteneurisation du DAG** (titre des issues 07-13) : remplacer les opérateurs
   par `make_container_task`. Léger, vérifiable par DagBag + lancement conteneur.
2. **Persistance runtime** (data plane) : implémenter l'écriture réelle dans les
   stores. Lourd, nécessite la logique métier dans les scripts.

Toutes les tranches DAG (07-13) sont désormais converties et chaque critère de
persistance/runtime réel a été isolé dans une issue sœur (07→18, 08→19, 09→20,
10→21, 11→22, 13→23 ; 12 close sans sœur, cf. ci-dessous). Le découpage
« conversion légère vs runtime lourd » est appliqué uniformément.

Cas particuliers :
- **Ingestion (07)** : conversion DAG faite, mais aucune issue ultérieure ne la
  revisite → la persistance est isolée dans la **nouvelle issue 18**.
- **Dataset (08)** : conversion DAG faite (lifecycle_decision + dataset en
  conteneurs) ; la matérialisation MinIO/PG du dataset candidat est isolée dans la
  **nouvelle issue 19** (même découpage que 07→18).
- **Train/eval (09)** : conversion DAG faite (train + eval en conteneurs ml, pool
  `iqa_gpu`, lock GPU) ; l'entraînement réel + matérialisation checkpoint/métriques
  MinIO est isolé dans la **nouvelle issue 20** (même découpage).
- **Gates/mlflow (10)** : conversion DAG faite. `gates` est **déjà réel et bloquant**
  (évalue `promotion_gates.yaml`, exit non-zéro si échec) ; seul l'enregistrement
  MLflow réel (le nom isolé par scénario est déjà réel) est isolé dans la **nouvelle
  issue 21** (même découpage).
- **Promotion/reload (11)** : conversion DAG faite (`promotion` sur image ml,
  `reload` sur image data) → DAG `iqa_lifecycle` 100 % conteneur, ADR 0008
  entièrement résolu. La regle `snapshot_previous_prod` (prod uniquement) et le
  skip non-prod du reload sont **déjà réels** ; la transition réelle au Registry
  et le reload HTTP de `iqa-inference` sont isolés dans la **nouvelle issue 22**
  (même découpage).
- **Replay (12)** : conversion DAG faite (BashOperator → `make_container_task`,
  image `data`, `iqa-run-replay`). Les critères portent sur la **sémantique** des
  événements rejoués (`event_time`/`recorded_at`/`is_simulated`) : vérifiable au
  niveau frontière (`preserved_event_fields`, validé sur le plan réel — 832
  événements) → **issue close sans sœur dédiée**. L'émission réelle des événements
  dans le store relève du runtime d'ingestion (issue 18, les rejoués transitent par
  l'ingestion), pas d'une nouvelle issue.
- **Monitoring (13)** : conversion DAG faite (BashOperator → `make_container_task`,
  image `data`, `iqa-run-monitoring` ; `drift_confirmed` passé en valeur). L'évaluation
  des seuils (`configs/monitoring_thresholds.yaml`) est **déjà réelle** dans le
  conteneur (`roi_fail_rate` comparé aux seuils warning/critical) ; seul le push des
  métriques vers Prometheus/Grafana (« métriques visibles dans Grafana ») est isolé
  dans la **nouvelle issue 23** (même découpage).

## Contrats transverses

A respecter par toutes les tranches DAG (07-13) :

- **Data lineage via stores, pas via XCom.** Chaque tache conteneur lit/ecrit ses
  donnees dans MinIO / PostgreSQL / MLflow. XCom ne transporte que des *references*
  (URI MinIO, `run_id` MLflow, `event_id`). C'est ce qui garde le lineage lisible et
  modulable : on remplace un conteneur sans casser les autres. Option future
  hors-scope : OpenLineage.
- **dev/prod par overlays Compose** (issue 17), pas un Dockerfile par service.
