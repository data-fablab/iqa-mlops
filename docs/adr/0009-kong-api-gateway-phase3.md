# ADR 0009 - Kong API Gateway for IQA Phase 3

## Status

Proposed for Phase 3 MVP.

## Context

IQA Phase 3 needs a public entry point for the application services.

The Gateway options considered for Phase 3 were NGINX, Traefik and Kong.

NGINX was considered as a simple reverse proxy option, but it is not retained for Phase 3.

Traefik was considered as a future Kubernetes oriented option, but it is not retained for Phase 3.

Kong is retained as the Phase 3 API Gateway MVP.

Phase 1 and Phase 2 already implemented application governance inside FastAPI.

Phase 3 adds an entry security layer in front of these application controls.

## Decision

Use Kong Gateway as the public entry point for IQA Phase 3.

Do not retain NGINX as the Phase 3 Gateway choice.

Do not retain Traefik for Phase 3.

FastAPI remains the IQA business API and application governance boundary.

Kong does not replace iqa-api.

Kong protects the entry.

FastAPI protects the business logic and AI governance.

## Target architecture

    user / demo / scripts / Airflow
            |
            v
    Kong Gateway
            |
            +--> iqa-api
            +--> iqa-streamlit
            +--> mlflow
            +--> minio console
            +--> grafana
            +--> airflow webserver

## Services exposed through Kong

| Kong public route | Upstream service | Upstream port |
| --- | --- | --- |
| /api/ | iqa-api | 8000 |
| /iqa/ | iqa-streamlit | 8501 |
| /mlflow/ | mlflow | 5000 |
| /minio/ | minio | 9001 |
| /grafana/ | grafana | 3000 |
| /airflow/ | airflow-webserver | 8080 |

## API route protection matrix

| Kong route | Upstream route | Protection |
| --- | --- | --- |
| GET /api/health | GET /health | public or low restriction |
| GET /api/model/version | GET /model/version | service authentication |
| POST /api/predict | POST /predict | service authentication, rate limit, request size limit, logs |
| POST /api/feedback | POST /feedback | service authentication, rate limit, request size limit, logs |
| GET /api/predictions | GET /predictions | service authentication |
| GET /api/lots/summary | GET /lots/summary | service authentication |
| GET /api/incidents | GET /incidents | security reviewer or admin authentication |
| GET /api/metrics | GET /metrics | monitoring only |
| POST /api/admin/reload-model | POST /admin/reload-model | admin authentication, strict rate limit, logs |

## Platform route protection matrix

| Kong route | Upstream service | Protection |
| --- | --- | --- |
| /iqa/ | iqa-streamlit | demo access control |
| /mlflow/ | mlflow | restricted admin access |
| /minio/ | minio console | restricted admin access |
| /grafana/ | grafana | restricted admin access |
| /airflow/ | airflow webserver | restricted admin access |

## Authentication strategy

Kong handles entry authentication.

FastAPI keeps its existing application security controls.

| Layer | Responsibility |
| --- | --- |
| Kong | entry authentication and consumer control |
| FastAPI | business contract, schema validation, service token and admin token |
| Metadata store | audit facts and runtime state |
| MLflow Registry | model source of truth |

Existing IQA tokens remain active.

| Token | Role |
| --- | --- |
| IQA_SERVICE_TOKEN | protects service level API calls |
| IQA_ADMIN_TOKEN | protects model reload |

## Rate limiting strategy

| Route class | Rate limit intent |
| --- | --- |
| /api/predict | protect inference capacity |
| /api/feedback | prevent feedback spam |
| /api/admin/reload-model | prevent reload abuse |
| /api/metrics | restrict scrape access |
| platform UIs | human access only |

Exact numeric limits are implemented in the next configuration task.

## Logging strategy

Kong must expose entry access logs.

| Event | Expected evidence |
| --- | --- |
| Accepted request | route, method, consumer, status |
| Refused request | route, method, reason, status |
| Rate limited request | route, consumer, status |
| Admin reload attempt | route, status |
| Monitoring access | source and status |

FastAPI incidents remain the source of truth for AI security events.

Kong logs are entry access evidence.

## Security headers and request limits

Kong should add or preserve security hardening at the entry point.

| Control | Intent |
| --- | --- |
| Security headers | reduce browser exposure |
| Request size limit | block oversized payloads |
| Rate limiting | reduce abuse |
| Authentication | restrict sensitive access |
| Access logs | support auditability |

## NGINX option

NGINX was considered as a simple reverse proxy option.

It is not retained as the Phase 3 API Gateway.

The Phase 3 architecture describes Kong as the public entry point.

## Traefik option

Traefik is not retained for Phase 3.

It remains a possible future option for Kubernetes oriented service discovery.

Kubernetes remains Phase 4, not Phase 3.

## Definition of done for IQA3_NAT02

| Item | Status |
| --- | --- |
| Kong target architecture is documented | required |
| Kong service routes are documented | required |
| API route protection matrix is documented | required |
| Authentication strategy is documented | required |
| Rate limiting strategy is documented | required |
| Logging strategy is documented | required |
| Security headers and request size limit are documented | required |
| NGINX option is documented as not retained | required |
| Traefik future position is documented | required |
| No server deployment is required | required |

## Follow up tasks

| Task | Purpose |
| --- | --- |
| IQA3_NAT03 | Create Kong declarative configuration |
| IQA3_NAT04 | Protect sensitive routes |
| IQA3_NAT05 | Add Gateway validation tests |
| IQA3_NAT06 | Document separation between API, metadata, artifacts and models |
