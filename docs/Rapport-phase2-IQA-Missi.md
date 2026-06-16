# Rapport Phase 2 - IQA MLOps

Ce document recapitule les travaux realises pour la Phase 2 (verrou GPU,
observabilite Prometheus/Grafana), en complement de `Architecture-Projet-IQA.md`
et du `Runbook-Phase1-IQA.md` (sections 6 et 7 mises a jour).

Etat de sortie : 95 tests verts, `ruff check` propre, `docker compose config`
valide.

## 1. Verrou GPU - pas de train concurrent pendant l'inference demo

Le serveur n'a qu'un GPU (RTX 3060). Au-dela du pool Airflow `iqa_gpu`
(1 slot) qui serialise les taches GPU *dans* Airflow, un verrou fichier
partage etend la garantie a tout processus de l'hote (ex. un `iqa-trainer`
lance a la main pendant une demo).

- `src/iqa/runtime/gpu_lock.py` : verrou `flock` cross-process via le context
  manager `gpu_lock(owner=..., blocking=...)`. Acquisition non bloquante par
  defaut (`GpuBusyError` si deja tenu), le fichier de verrou enregistre le
  detenteur. Chemin configurable par `IQA_GPU_LOCK_PATH`.
- `src/iqa/runtime/__init__.py` : expose l'API publique du module runtime.
- `scripts/run_lifecycle.py` : le trainer prend le verrou avant de tourner ;
  refus immediat avec sortie 75 si occupe. Options `--wait-for-gpu` (attendre
  la liberation) et `--no-gpu-lock` (dry-run CPU).
- `src/iqa/inference/service.py` : via un `lifespan` FastAPI, la demo prend le
  verrou au demarrage et le garde tant qu'elle tourne quand `IQA_GPU_DEMO_HOLD`
  est actif. Nouvelle metrique `iqa_inference_gpu_lock_held`.
- `deploy/docker-compose.yml` : volume nomme `gpu_lock` partage entre
  `iqa-inference` et `iqa-trainer`, variables `IQA_GPU_LOCK_PATH` /
  `IQA_GPU_DEMO_HOLD`.
- `tests/test_gpu_lock.py` : trois tests (detenteur enregistre, refus en cas de
  detention concurrente, reutilisable apres liberation).

## 2. Prometheus - scrape API, Airflow et services

- `deploy/prometheus/prometheus.yml` : jobs `prometheus` (self), `iqa-api`,
  `iqa-inference`, `airflow` (via `statsd-exporter:9102`) et `minio`.
- `deploy/docker-compose.yml` :
  - service `statsd-exporter` (`prom/statsd-exporter`) qui convertit le StatsD
    Airflow en metriques Prometheus ;
  - `airflow-webserver` et `airflow-scheduler` configures en StatsD
    (`AIRFLOW__METRICS__STATSD_*`) ;
  - `minio` avec `MINIO_PROMETHEUS_AUTH_TYPE=public` pour exposer ses metriques
    cluster sans authentification ;
  - `prometheus` depend de `statsd-exporter`.

## 3. Dashboard Grafana minimal

Provisioning automatique (le dossier `provisioning/` est monte dans le service
`grafana`) :

- `deploy/grafana/provisioning/datasources/prometheus.yml` : datasource
  Prometheus (uid `prometheus`, `http://prometheus:9090`).
- `deploy/grafana/provisioning/dashboards/dashboards.yml` : provider fichier
  qui charge les JSON depuis `./json` (dossier "IQA").
- `deploy/grafana/provisioning/dashboards/json/iqa-overview.json` : dashboard
  `IQA - Vue d'ensemble`.

Panneaux livres :

- **Modele actif** : `iqa_active_model_info` (versions Feature-AE + ROI).
- **Distribution V/O/R** : `iqa_prediction_total{decision=...}`.
- **Latence predict** : `iqa_predict_latency_seconds_{sum,count}`.
- **Erreurs** : `iqa_invalid_feedback_total`, `iqa_reload_refused_total`.
- **ROI fail** : `iqa_roi_fail_total`.
- **Incidents IA** : `iqa_ai_security_incident_total`,
  `iqa_feedback_conflict_total`, `iqa_unsafe_train_blocked_total`.
- **Disponibilite / GPU** : `iqa_api_up`, `iqa_inference_up`,
  `iqa_inference_gpu_lock_held`.

## 4. Metriques API ajoutees

`src/iqa/api/main.py` expose desormais, sur `/metrics`, des metriques reelles
alimentees par le chemin `/predict` :

- `iqa_prediction_total{decision="Vert|Orange|Rouge"}` (compteurs V/O/R) ;
- `iqa_roi_fail_total` (echecs ROI au moment du predict) ;
- `iqa_predict_latency_seconds_sum` / `_count` (latence, moyenne par taux) ;
- `iqa_active_model_info{feature_ae_version=...,roi_model_version=...}` (modele
  actif, lu depuis les manifests).

Verifie de bout en bout : un appel `/predict` incremente bien
`iqa_prediction_total{decision="Vert"}` et renseigne `iqa_active_model_info`.

## 5. Documentation mise a jour

- `docs/Runbook-Phase1-IQA.md` : section 6 (cibles Prometheus + dashboard
  Grafana) et section 7 (sous-section "Verrou GPU" pour la demo).
- `deploy/grafana/provisioning/README.md` : layout reel du provisioning et liste
  des panneaux/metriques.
- `.env.example` : variables `IQA_GPU_LOCK_PATH` et `IQA_GPU_DEMO_HOLD`.

## 6. Etat des taches Phase 2

- Verrou GPU (pas de train concurrent pendant inference demo) : fait.
- Nginx (`/IQA`, `/api`, `/mlflow`, `/minio`, `/grafana`, `/airflow`) : deja
  livre en Phase 1.
- Prometheus scrape API, Airflow et services : fait.
- Dashboard Grafana minimal (V/O/R, latence, erreurs, ROI fail, incidents IA,
  modele actif) : fait.

## 7. Fichiers touches

Nouveaux :

```text
src/iqa/runtime/__init__.py
src/iqa/runtime/gpu_lock.py
tests/test_gpu_lock.py
deploy/grafana/provisioning/datasources/prometheus.yml
deploy/grafana/provisioning/dashboards/dashboards.yml
deploy/grafana/provisioning/dashboards/json/iqa-overview.json
```

Modifies :

```text
.env.example
deploy/docker-compose.yml
deploy/prometheus/prometheus.yml
deploy/grafana/provisioning/README.md
docs/Runbook-Phase1-IQA.md
scripts/run_lifecycle.py
src/iqa/api/main.py
src/iqa/inference/service.py
```
