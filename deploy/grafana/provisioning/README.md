# Grafana provisioning

This directory is mounted into the `grafana` service at
`/etc/grafana/provisioning` (see `deploy/docker-compose.yml`).

Layout expected by Grafana's provisioning system:

```text
provisioning/
|-- datasources/   -> Prometheus datasource definitions (YAML)
`-- dashboards/    -> dashboard provider config + dashboard JSON
```

Dashboards for Marc (IQA monitoring: drift, latency, Vert/Orange/Rouge
distribution) will be added here as the monitoring DAG (`iqa_monitoring.py`)
produces the corresponding metrics.
