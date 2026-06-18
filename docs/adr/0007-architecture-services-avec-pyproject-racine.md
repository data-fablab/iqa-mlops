# ADR 0007 - Services Docker avec `pyproject.toml` racine conserve

## Statut

Accepte. Amende le 2026-06-17 : le `pyproject.toml` racine est conserve, mais les dependances et les images sont desormais decoupees par service (voir section "Amendement").

## Contexte

La proposition de convergence recommande une isolation forte des runtimes. Elle signale aussi qu'Airflow et l'API peuvent avoir des contraintes de dependances incompatibles.

Le repo IQA actuel dispose deja d'un package `src/iqa`, d'un `pyproject.toml` racine, de commandes publiques et de tests reproductibles via `uv`.

## Decision

On conserve le pyproject.toml racine en phase initiale.

L'architecture cible reste microservices Docker :

```text
iqa-api
iqa-inference
iqa-ingestion
iqa-replay
iqa-trainer
iqa-monitoring
airflow
mlflow
minio
postgres
prometheus
grafana
nginx
```

`iqa-api` et `iqa-inference` restent separes dans l'architecture cible. Airflow orchestre par contrats HTTP ou commandes batch, et n'a pas vocation a importer le runtime API/inference.

Une migration vers un dossier `services/` avec un environnement par service reste possible plus tard, mais elle est hors scope de cette passe.

## Consequences

Le repo reste simple a installer :

```powershell
uv sync --extra cpu
uv run --extra cpu pytest -q
```

L'isolation fine se fait d'abord par Docker Compose et par les frontieres de services, pas par une refonte immediate du layout.

## Amendement (2026-06-17) - Decoupage des dependances et des images par service

### Contexte

L'objectif du projet est devenu "le plus microservice possible". Or, en pratique,
toutes les dependances (y compris `torch`/`torchvision`) sont dans le bloc
`dependencies` de base du `pyproject.toml`. Consequence : `iqa-api` et
`iqa-ingestion`, qui n'utilisent pas PyTorch, embarquent quand meme le runtime
GPU. Au niveau deploiement les services sont bien separes (un conteneur par
service), mais au niveau image/dependances ils restent monolithiques.

### Decision

On conserve le mono-repo et le `pyproject.toml` racine (faible friction
d'installation), mais on le deplie en extras par role :

```text
serving  -> fastapi, uvicorn, pydantic, boto3 (sans torch)   [iqa-api]
ml       -> torch, torchvision, scikit-learn, mlflow          [iqa-inference, iqa-trainer]
data     -> pandas, pillow, boto3, psycopg                    [iqa-ingestion, iqa-replay, iqa-monitoring]
```

Le `Dockerfile` devient multi-stage : un stage par extra, produisant des images
distinctes et slim. `iqa-api` ne contient plus PyTorch.

### Consequences

- Vraie isolation des runtimes : images plus petites, blast radius reduit.
- Chaque image correspond naturellement a un pod Kubernetes (voir [ADR 0008](0008-taches-airflow-comme-conteneurs.md)).
- Le decoupage en dossier `services/` reste hors scope : on isole par extras et
  par stages Docker, pas par split du repo.
- `torch>=2.1`/`torchvision>=0.16` quittent `dependencies` de base pour l'extra `ml`.
