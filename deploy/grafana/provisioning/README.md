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
        |-- iqa-overview.json          -> IQA operational overview dashboard
        |-- iqa-executive-mlops.json   -> narrative executive MLOps dashboard
        |-- iqa-lifecycle.json         -> lifecycle learning dashboard
        `-- iqa-drift-p4.json          -> Piece A/P4 drift correction dashboard
```

The provisioned dashboards are split by audience and demo moment:

- **IQA - Vue d'ensemble** (`iqa-overview`) keeps the minimal operational MVP
  view for API, inference, predictions, latency, ROI failures, and security
  counters.
- **IQA - Vue Executive MLOps** (`iqa-executive-mlops`) opens the demo with
  the story: start small, learn under control, observe, confirm drift, correct
  with traceability.
- **IQA - Lifecycle MLOps** (`iqa-lifecycle`) supports scenario 1, the first
  week of controlled learning on Piece B.
- **IQA - Drift P4** (`iqa-drift-p4`) supports scenario 2, Piece A/P4 drift
  detection and targeted correction.

The `iqa-overview` dashboard gives the minimal MVP view:

- **Modele actif** : `iqa_active_model_info` (Feature-AE + ROI segmenter versions).
- **Distribution V/O/R** : `iqa_prediction_total{decision=...}`.
- **Latence predict** : `iqa_predict_latency_seconds_{sum,count}`.
- **Erreurs** : `iqa_invalid_feedback_total`, `iqa_reload_refused_total`.
- **ROI fail** : `iqa_roi_fail_total`.
- **Incidents IA** : `iqa_ai_security_incident_total`,
  `iqa_feedback_conflict_total`, `iqa_unsafe_train_blocked_total`.
- **Disponibilite / GPU** : `iqa_api_up`, `iqa_inference_up`,
  `iqa_inference_gpu_lock_held`.

The narrative dashboards use only already exposed API/inference `/metrics`
series scraped by Prometheus (`deploy/prometheus/prometheus.yml`): lifecycle
run summaries, epoch metrics, gate decisions, final model info, drift status,
window indexes, domain ratios, and correction triggers. They expose
operational signals and model registry labels, not industrial image paths,
masks, local files, or cache artifacts. The runtime source of truth is the JSON
in this provisioning directory.
