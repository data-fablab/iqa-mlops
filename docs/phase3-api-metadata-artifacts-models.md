# IQA Phase 3 Separation Between API, Metadata, Artifacts and Models

## Purpose

This document explains the separation between the Gateway, the API, metadata, artifacts, models, metrics and orchestration in IQA Phase 3.

## Responsibility matrix

| Layer | Responsibility |
| --- | --- |
| Kong Gateway | public entry point, routing, authentication, rate limiting and access logs |
| FastAPI iqa-api | business contracts, schemas, prediction, feedback, incidents and audit rules |
| PostgreSQL metadata store | structured runtime facts and audit state |
| MinIO | heavy files, images, masks, heatmaps and artifacts |
| MLflow Registry | model version source of truth |
| Prometheus and Grafana | metrics and dashboards |
| Airflow | lifecycle orchestration |
| DVC | dataset and replay lineage |

## API boundary

FastAPI remains the application governance boundary.

It validates schemas, enforces scenario_id, checks feedback consistency, preserves oracle_gt sovereignty, keeps human_sophie display only, creates incidents and exports metrics.

Kong does not replace FastAPI.

Kong protects the entry.

FastAPI protects the business logic and AI governance.

## Metadata boundary

PostgreSQL stores structured runtime metadata.

Examples include prediction_id, piece_event_id, scenario_id, decisions, feedback state, reload events, incident events and active model runtime facts.

PostgreSQL does not store heavy artifacts.

## Artifact boundary

MinIO stores heavy files.

Examples include source images, ROI masks, heatmaps, replay outputs and ML artifacts.

The API stores artifact references, not the binary files.

The metadata store keeps URIs, hashes and runtime facts.

## Model boundary

MLflow Registry is the model source of truth.

It stores registered model identity, model versions, aliases, stages, run links and artifact links.

FastAPI must not invent model versions.

FastAPI reads model state from MLflow Registry contracts.

## Lineage boundary

DVC and manifests keep dataset and replay lineage.

Important lineage fields include dataset_version, manifest_version, git_commit, MLflow run id and MinIO artifact URI.

This explains which data produced which model and which model served which prediction.

## End to end flow

User or Airflow calls Kong.

Kong forwards authorized requests to FastAPI.

FastAPI writes structured facts to PostgreSQL.

FastAPI stores or references heavy artifacts in MinIO.

FastAPI reads model truth from MLflow Registry.

Metrics are exposed to Prometheus and displayed in Grafana.

## Security interpretation

Laurent can audit entry access in Kong logs.

Laurent can audit business decisions in FastAPI contracts.

Laurent can audit runtime facts in PostgreSQL.

Laurent can audit artifacts through MinIO URIs and hashes.

Laurent can audit model versions through MLflow Registry.

Laurent can audit data lineage through DVC and manifests.

## NAT06 validation

| Requirement | Status |
| --- | --- |
| API boundary documented | yes |
| metadata boundary documented | yes |
| artifact boundary documented | yes |
| model boundary documented | yes |
| lineage boundary documented | yes |
| security interpretation documented | yes |
