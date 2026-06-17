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

## Chemin critique

```text
01 -> 02 -> 05 -> 06 -> 08 -> 09 -> 10 -> 11 -> 16
```

Les images (03/04) et la CI (14 -> 15) se parallelisent ; 00 (HITL) doit etre tranchee avant 14.

## Lots de travail

- Microservices/images : 01, 02, 03, 04, 17
- Orchestration conteneurisee : 05, 06, 07, 08, 09, 10, 11, 12, 13
- Automatisation / registre : 00, 14, 15, 16

## Contrats transverses

A respecter par toutes les tranches DAG (07-13) :

- **Data lineage via stores, pas via XCom.** Chaque tache conteneur lit/ecrit ses
  donnees dans MinIO / PostgreSQL / MLflow. XCom ne transporte que des *references*
  (URI MinIO, `run_id` MLflow, `event_id`). C'est ce qui garde le lineage lisible et
  modulable : on remplace un conteneur sans casser les autres. Option future
  hors-scope : OpenLineage.
- **dev/prod par overlays Compose** (issue 17), pas un Dockerfile par service.
