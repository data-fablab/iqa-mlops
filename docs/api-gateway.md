# IQA Phase 3 Kong Gateway Design

## Purpose

This document defines the IQA Phase 3 Gateway design.

The selected Gateway is Kong.

Kong becomes the public entry point for IQA services.

NGINX is not retained as the Phase 3 Gateway.

Traefik is not retained for Phase 3.

## Core principle

Kong protects the entry.

FastAPI protects the business logic and AI governance.

## Target flow

    user / demo / scripts / Airflow
            |
            v
    Kong Gateway
            |
            +--> iqa-api
            +--> iqa-streamlit
            +--> MLflow
            +--> MinIO
            +--> Grafana
            +--> Airflow

## Services exposed through Kong

| Public route | Upstream service | Purpose |
| --- | --- | --- |
| /api/ | iqa-api:8000 | IQA API contracts |
| /iqa/ | iqa-streamlit:8501 | Sophie review interface |
| /mlflow/ | mlflow:5000 | model registry and runs |
| /minio/ | minio:9001 | artifact console |
| /grafana/ | grafana:3000 | monitoring dashboards |
| /airflow/ | airflow-webserver:8080 | orchestration UI |

## API routes

| External route | Upstream route | Protection |
| --- | --- | --- |
| GET /api/health | GET /health | low restriction |
| GET /api/model/version | GET /model/version | service authentication |
| POST /api/predict | POST /predict | service authentication, rate limit, logs |
| POST /api/feedback | POST /feedback | service authentication, rate limit, logs |
| GET /api/predictions | GET /predictions | service authentication |
| GET /api/lots/summary | GET /lots/summary | service authentication |
| GET /api/incidents | GET /incidents | security reviewer or admin authentication |
| GET /api/metrics | GET /metrics | monitoring only |
| POST /api/admin/reload-model | POST /admin/reload-model | admin authentication, strict rate limit, logs |

## Authentication design

Kong authenticates consumers at the entry point.

FastAPI keeps the existing application tokens.

| Route class | Kong control | FastAPI control |
| --- | --- | --- |
| prediction | service key | IQA_SERVICE_TOKEN |
| feedback | service key | feedback contract checks |
| incidents | security reviewer key | incident contract |
| metrics | monitoring key | metrics contract |
| admin reload | admin key | IQA_ADMIN_TOKEN |

## Rate limiting design

| Route class | Intent |
| --- | --- |
| /api/predict | protect inference |
| /api/feedback | reduce spam and poisoning attempts |
| /api/admin/reload-model | prevent reload abuse |
| /api/metrics | restrict monitoring exposure |
| admin UIs | human admin access only |

## Logging design

Kong access logs must support security review.

| Log need | Example evidence |
| --- | --- |
| accepted requests | consumer, route, method, status |
| refused requests | route, method, status |
| rate limited requests | consumer, route, status |
| admin reload attempts | consumer, status |
| monitoring access | source and status |

FastAPI incidents remain application level AI security evidence.

Kong logs remain entry level security evidence.

## Security headers and request size limit

Kong should add or preserve security headers at the entry point.

Kong should enforce request size limits on sensitive API routes.

| Control | Purpose |
| --- | --- |
| Security headers | reduce browser exposure |
| Request size limit | reject oversized payloads |
| Rate limiting | reduce abuse |
| Authentication | restrict sensitive access |
| Access logs | support auditability |

## NGINX option

NGINX was considered as a simple reverse proxy option.

It is not retained for Phase 3.

The Phase 3 Gateway is Kong.

## Traefik option

Traefik is not selected for Phase 3.

Traefik can be reconsidered later for Kubernetes oriented service discovery.

## NAT02 validation

| Requirement | Documented |
| --- | --- |
| Kong target Gateway | yes |
| Routes and services | yes |
| Authentication | yes |
| Rate limiting | yes |
| Logs | yes |
| Security headers and request size limit | yes |
| NGINX option not retained | yes |
| Traefik future note | yes |
