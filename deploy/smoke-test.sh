#!/usr/bin/env bash
# Smoke tests deploiement IQA.
#
# Verifie qu'un stack docker-compose demarre repond correctement : sante des
# services, endpoints API cles et back-ends d'observabilite. A lancer apres
# `docker compose up`. Codes de sortie : 0 = tout vert, 1 = au moins un echec.
#
# Hotes configurables (defaut = ports publies en local) :
#   IQA_API_URL=http://localhost:8000
#   IQA_INFERENCE_URL=http://localhost:8100
#   IQA_MINIO_URL=http://localhost:9000
#   IQA_PROMETHEUS_URL=http://localhost:9090
#   IQA_GRAFANA_URL=http://localhost:3000
#   IQA_MLFLOW_URL=http://localhost:5000
#   IQA_AIRFLOW_URL=http://localhost:8080
#   IQA_GATEWAY_URL=http://localhost      (reverse-proxy, port 80)
set -u

API_URL="${IQA_API_URL:-http://localhost:8000}"
INFERENCE_URL="${IQA_INFERENCE_URL:-http://localhost:8100}"
MINIO_URL="${IQA_MINIO_URL:-http://localhost:9000}"
PROMETHEUS_URL="${IQA_PROMETHEUS_URL:-http://localhost:9090}"
GRAFANA_URL="${IQA_GRAFANA_URL:-http://localhost:3000}"
MLFLOW_URL="${IQA_MLFLOW_URL:-http://localhost:5000}"
AIRFLOW_URL="${IQA_AIRFLOW_URL:-http://localhost:8080}"
GATEWAY_URL="${IQA_GATEWAY_URL:-http://localhost}"

fail=0

# check NAME URL [EXPECTED_SUBSTRING]
check() {
  name="$1"
  url="$2"
  expected="${3:-}"
  body="$(curl -fsS --max-time 10 "$url" 2>/dev/null)"
  status=$?
  if [ $status -ne 0 ]; then
    echo "FAIL  $name -> $url (injoignable)"
    fail=1
    return
  fi
  if [ -n "$expected" ] && ! printf '%s' "$body" | grep -q "$expected"; then
    echo "FAIL  $name -> $url (reponse inattendue, attendu: '$expected')"
    fail=1
    return
  fi
  echo "OK    $name"
}

echo "== Smoke tests IQA =="

# Services applicatifs
check "api /health"             "$API_URL/health"            '"status":"ok"'
check "inference /health"       "$INFERENCE_URL/health"      '"status":"ok"'
check "api /metrics"            "$API_URL/metrics"           "iqa_api_up 1"
check "inference /metrics"      "$INFERENCE_URL/metrics"     "iqa_inference_up 1"
check "api /model/version"      "$API_URL/model/version"     "feature_ae"
check "api /replay-scenarios"   "$API_URL/replay-scenarios"
check "api /predictions"        "$API_URL/predictions"
check "api /lots/summary"       "$API_URL/lots/summary"

# Back-ends infra / observabilite
check "minio live"              "$MINIO_URL/minio/health/live"
check "prometheus healthy"      "$PROMETHEUS_URL/-/healthy"
check "prometheus targets"      "$PROMETHEUS_URL/api/v1/targets" '"status":"success"'
check "grafana health"          "$GRAFANA_URL/api/health"    '"database"'
check "mlflow up"               "$MLFLOW_URL/"

# Orchestration & monitoring
# Airflow expose /health (metadatabase + scheduler). iqa-monitoring est un job
# batch (profile "batch", sans HTTP) : sa couverture passe par Airflow + les
# metriques Prometheus (job "airflow" via statsd-exporter) ci-dessus.
check "airflow health"          "$AIRFLOW_URL/health"        "healthy"

# Gateway (reverse-proxy) : valide le routage de bout en bout sur le port 80.
check "gateway -> api"          "$GATEWAY_URL/api/health"        '"status":"ok"'
check "gateway -> grafana"      "$GATEWAY_URL/grafana/api/health" '"database"'
check "gateway -> airflow"      "$GATEWAY_URL/airflow/health"    "healthy"
check "gateway -> mlflow"       "$GATEWAY_URL/mlflow/"

echo "===================="
if [ "$fail" -eq 0 ]; then
  echo "RESULTAT : tous les smoke tests sont verts."
else
  echo "RESULTAT : au moins un smoke test a echoue."
fi
exit "$fail"
