# Grafana provisioning

This directory is mounted into the `grafana` service at
`/etc/grafana/provisioning` (see `deploy/docker-compose.yml`).

Layout:

```text
provisioning/
|-- datasources/
|   `-- prometheus.yml         -> Prometheus datasource (uid: prometheus)
`-- dashboards/
    |-- dashboards.yml         -> file provider (loads JSON from ./json)
    `-- json/
        `-- iqa-overview.json  -> IQA overview dashboard
```

The `iqa-overview` dashboard (folder "IQA") gives the minimal MVP view:

- **Modele actif** : `iqa_active_model_info` (Feature-AE + ROI segmenter versions).
- **Distribution V/O/R** : `iqa_prediction_total{decision=...}`.
- **Latence predict** : `iqa_predict_latency_seconds_{sum,count}`.
- **Erreurs** : `iqa_invalid_feedback_total`, `iqa_reload_refused_total`.
- **ROI fail** : `iqa_roi_fail_total`.
- **Incidents IA** : `iqa_ai_security_incident_total`,
  `iqa_feedback_conflict_total`, `iqa_unsafe_train_blocked_total`.
- **Disponibilite / GPU** : `iqa_api_up`, `iqa_inference_up`,
  `iqa_inference_gpu_lock_held`.

All metrics come from the API/inference `/metrics` endpoints scraped by
Prometheus (`deploy/prometheus/prometheus.yml`). Decision/latency/ROI counters
are fed by the `/predict` path; richer drift series will be added as the
monitoring DAG (`iqa_monitoring.py`) emits them.
