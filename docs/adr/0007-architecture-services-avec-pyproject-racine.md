# ADR 0007 - Services Docker avec `pyproject.toml` racine conserve

## Statut

Accepte.

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
