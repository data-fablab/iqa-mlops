#!/usr/bin/env bash
# Demo IQA "from scratch" : repart d'un etat vierge et deroule la chaine
# complete pour une soutenance / validation reproductible.
#
#   1. docker compose down -v        (DESTRUCTIF : efface les volumes)
#   2. docker compose up -d --build  (socle + app + observabilite + gateway)
#   3. attente de disponibilite de l'API
#   4. deploy/smoke-test.sh          (sante de toute la stack)
#   5. iqa-demo-phase2               (predict -> feedback -> Marc/Sophie)
#
# Usage : bash deploy/demo-from-scratch.sh [--yes]
#   --yes : ne pas demander confirmation avant le `down -v` destructif.
set -euo pipefail

ASSUME_YES=0
[ "${1:-}" = "--yes" ] && ASSUME_YES=1

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEPLOY="$ROOT/deploy"
API_URL="${IQA_API_URL:-http://localhost:8000}"

echo "== Demo IQA from scratch =="

# --- .env requis par docker compose -----------------------------------------
if [ ! -f "$ROOT/.env" ]; then
  echo "INFO  .env absent -> copie depuis .env.example (pense a aligner les creds MinIO)."
  cp "$ROOT/.env.example" "$ROOT/.env"
fi

# --- 1. etat vierge (destructif) --------------------------------------------
if [ "$ASSUME_YES" -ne 1 ]; then
  printf "ATTENTION: 'docker compose down -v' va EFFACER les volumes (postgres, minio). Continuer ? [y/N] "
  read -r answer
  case "$answer" in
    [yY]|[yY][eE][sS]) ;;
    *) echo "Abandon."; exit 1 ;;
  esac
fi

cd "$DEPLOY"
echo "-- 1/5 down -v (etat vierge)"
docker compose down -v --remove-orphans || true

# --- 2. demarrage de la stack -----------------------------------------------
echo "-- 2/5 up -d --build (stack de demo)"
docker compose up -d --build

# --- 3. attente API ----------------------------------------------------------
echo "-- 3/5 attente de l'API ($API_URL/health)"
ready=0
for _ in $(seq 1 60); do
  if curl -fsS --max-time 5 "$API_URL/health" >/dev/null 2>&1; then ready=1; break; fi
  sleep 5
done
if [ "$ready" -ne 1 ]; then
  echo "FAIL  API non disponible apres 5 min. Voir 'docker compose logs'."
  exit 1
fi
echo "OK    API disponible. (Airflow peut mettre encore quelques instants.)"
sleep 15

# --- 4. smoke test -----------------------------------------------------------
echo "-- 4/5 smoke test deploiement"
cd "$ROOT"
bash deploy/smoke-test.sh

# --- 5. parcours metier de demo ---------------------------------------------
echo "-- 5/5 parcours de demo (predict -> feedback -> Marc/Sophie)"
if command -v uv >/dev/null 2>&1; then
  uv run --extra cpu --extra serving iqa-demo-phase2
else
  echo "INFO  uv absent : lance le parcours manuellement -> 'uv run iqa-demo-phase2'."
fi

echo "== Demo prete =="
echo "  Streamlit (Accueil / Marc / Sophie) : http://localhost:8501"
echo "  Grafana 'IQA - Vue d'ensemble'       : http://localhost:3000"
echo "  Arret : (cd deploy && docker compose down)   # -v pour tout effacer"
