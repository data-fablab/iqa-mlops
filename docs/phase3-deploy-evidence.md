# Phase 3 Deploy Evidence

This document proves that IQA Phase 3 can be deployed from published images
instead of rebuilding application images on the server.

## Deployment Contract

Production deployment uses:

```bash
cd deploy
docker compose -f docker-compose.yml -f docker-compose.prod.yml pull
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
bash ../deploy/smoke-test.sh
```

The prod overlay uses `IQA_IMAGE_REGISTRY` and `IQA_IMAGE_TAG` for every IQA
application image. The recommended proof tag is the immutable GitHub Actions
SHA tag, for example `IQA_IMAGE_TAG=sha-<commit>`. Release tags such as
`v0.1.0` are valid only after a matching Git tag has been pushed and published
by CI. `IQA_IMAGE_TAG` must never be `latest`.

Published images:

- `iqa-serving` for `iqa-api`;
- `iqa-ml` for `iqa-inference` and `iqa-trainer`;
- `iqa-data` for ingestion, replay and monitoring jobs;
- `iqa-dvc-gate` for the Airflow/DVC reproducibility gate;
- `iqa-airflow` for Airflow webserver, scheduler and init.

The prod overlay explicitly disables inherited `build:` fallbacks for these
published images. Local builds remain a development concern only.

## Gateway And Proxy Roles

Kong is the Phase 3 API Gateway / policy layer target. It is responsible for
cross-cutting API policies such as route protection, authentication extension,
rate limiting and audit-oriented routing.

Nginx remains the current reverse-proxy fallback used by the compose stack and
smoke tests. It exposes stable paths for API, Streamlit, MLflow, MinIO, Grafana
and Airflow while Kong is progressively integrated.

The two are not the same responsibility:

- `iqa-api` is the business API;
- Kong is the API Gateway / policy layer;
- Nginx is the pragmatic reverse proxy fallback.

## CI Evidence

The GitHub Actions `publish-images` job builds and pushes immutable Docker Hub
images. It is opt-in: if the repository variables/secrets below are missing,
the job is skipped and the server will not be able to pull IQA images.

Repository variables:

```bash
IQA_PUBLISH_IMAGES=true
IQA_IMAGE_REGISTRY=<namespace-dockerhub>
```

Repository secrets:

```bash
DOCKERHUB_USERNAME
DOCKERHUB_TOKEN
```

`DOCKERHUB_TOKEN` must be a Docker Hub access token with write permission.
The workflow explicitly disables floating `latest` tags.

If you have repository admin rights, the prerequisites can be set with:

```bash
gh variable list
gh secret list
gh variable set IQA_PUBLISH_IMAGES --body true
gh variable set IQA_IMAGE_REGISTRY --body <namespace-dockerhub>
gh secret set DOCKERHUB_USERNAME --body <user>
gh secret set DOCKERHUB_TOKEN
```

The published image set covers application roles and the custom Airflow image,
so the server deployment path can be `pull -> up -d -> smoke` without rebuilding
IQA services locally.

Docker Hub is accessed by the server with `docker login` and HTTPS registry
pulls. SSH is not part of Docker Hub authentication; SSH is only needed later if
GitHub Actions is extended to trigger a remote deployment command on the server.

Before running the full compose deployment, the server proof should validate the
five IQA image pulls explicitly:

```bash
docker login
docker pull <namespace-dockerhub>/iqa-airflow:sha-<commit>
docker pull <namespace-dockerhub>/iqa-serving:sha-<commit>
docker pull <namespace-dockerhub>/iqa-ml:sha-<commit>
docker pull <namespace-dockerhub>/iqa-data:sha-<commit>
docker pull <namespace-dockerhub>/iqa-dvc-gate:sha-<commit>
```

## Smoke Coverage

`deploy/smoke-test.sh` validates:

- API and inference health/metrics;
- API model, replay, prediction and lot summary surfaces;
- MinIO, MLflow, Prometheus and Grafana;
- Airflow health;
- gateway routing to API, Grafana, Airflow and MLflow.

The smoke test is a deployment proof, not a model lifecycle trigger. MLflow
Registry remains the model source of truth, and model rollback is a Registry
operation, not a container redeploy.

## Static Evidence Command

Run:

```bash
uv run --extra cpu iqa-check-deploy-evidence
```

The command verifies that prod images are tag-controlled, `latest` is absent,
published services are declared, smoke coverage is present, and deployment docs
describe Docker Hub, Kong, Nginx fallback and MLflow Registry separation.
