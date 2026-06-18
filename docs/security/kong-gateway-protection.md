# IQA Phase 3 Kong Gateway Protection

## Purpose

This document records the sensitive route protection implemented for IQA Phase 3 through Kong Gateway.

## Protected sensitive routes

| Route | Purpose | Kong protection |
| --- | --- | --- |
| /api/admin/reload-model | model reload | key auth, admin ACL, strict rate limit, request size limit |
| /api/metrics | Prometheus metrics | key auth, monitoring or admin ACL, rate limit |
| /mlflow | model registry UI | key auth, admin ACL |
| /minio | artifact console | key auth, admin ACL |
| /grafana | monitoring dashboards | key auth, admin or monitoring ACL |
| /airflow | orchestration UI | key auth, admin ACL |

## Defense in depth

Kong protects entry access.

FastAPI remains responsible for business validation, AI governance, application tokens, audit trail and incidents.

## NAT04 validation evidence

The Kong declarative configuration contains:

| Control | Status |
| --- | --- |
| admin reload route | OK |
| metrics route | OK |
| MLflow route | OK |
| Grafana route | OK |
| Airflow route | OK |
| MinIO route | OK |
| key authentication | OK |
| ACL groups | OK |
| rate limiting | OK |
| request size limiting | OK |

## Scope limit

This document validates static route protection for IQA3_NAT04.

Executable Gateway tests are handled separately in IQA3_NAT05.
