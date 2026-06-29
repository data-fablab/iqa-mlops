#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_iqa_piece_a_p4_drift_trigger.sh --phase clear|suspected|confirmed [options]

Purpose:
  Smoke-test iqa_lifecycle_trigger with synthetic Piece B -> Piece A/P4 drift
  metrics. The natural scenario is iqa_drift_piece_a_p4: it runs inference
  first and derives drift from observed replay metrics.

Options:
  --phase VALUE                 clear, suspected, or confirmed.
  --epochs VALUE                Correction epochs if drift is confirmed. Default: 16.
  --allow-correction-trigger    Required for --phase confirmed, because it can launch training.
  --dry-run                     Write and print the Airflow conf only.
  --skip-checks                 Skip Docker/Airflow preflight checks.
  --keep-trigger-unpaused       Do not pause iqa_lifecycle_trigger after clear/suspected tests.
  -h, --help
USAGE
}

log_cmd() {
  printf '>>'
  printf ' %q' "$@"
  printf '\n'
}

run_checked() {
  log_cmd "$@"
  "$@"
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DEPLOY_DIR="$REPO_ROOT/deploy"
ENV_FILE="$REPO_ROOT/.env"
CACHE_DIR="$REPO_ROOT/.cache/iqa/drift_triggers"
ORIGINAL_DIR="$(pwd)"
trap 'cd "$ORIGINAL_DIR" >/dev/null' EXIT

case "${OSTYPE:-}" in
  msys*|cygwin*)
    export MSYS_NO_PATHCONV=1
    export MSYS2_ARG_CONV_EXCL='*'
    export MSYS2_ENV_CONV_EXCL='*'
    ;;
esac

PHASE=""
EPOCHS="16"
ALLOW_CORRECTION_TRIGGER="false"
DRY_RUN="false"
SKIP_CHECKS="false"
KEEP_TRIGGER_UNPAUSED="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase) PHASE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --allow-correction-trigger) ALLOW_CORRECTION_TRIGGER="true"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --skip-checks) SKIP_CHECKS="true"; shift ;;
    --keep-trigger-unpaused) KEEP_TRIGGER_UNPAUSED="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$PHASE" in
  clear|suspected|confirmed) ;;
  *) printf 'Missing or invalid --phase. Expected clear, suspected, or confirmed.\n\n' >&2; usage >&2; exit 2 ;;
esac

if [[ "$PHASE" == "confirmed" && "$ALLOW_CORRECTION_TRIGGER" != "true" && "$DRY_RUN" != "true" ]]; then
  cat >&2 <<'EOF'
Refusing to run phase=confirmed without --allow-correction-trigger.

This phase is expected to trigger iqa_lifecycle and start correction training.
Rerun with --dry-run to inspect the conf, or add --allow-correction-trigger
when you really want to validate the retraining trigger.
EOF
  exit 2
fi

if [[ ! -f "$ENV_FILE" ]]; then
  printf 'Missing env file: %s\n' "$ENV_FILE" >&2
  exit 1
fi

mkdir -p "$CACHE_DIR"
CONF_PATH="$CACHE_DIR/iqa_lifecycle_trigger_piece_a_p4_${PHASE}.json"
PY_CONF_PATH="$CONF_PATH"
if command -v cygpath >/dev/null 2>&1; then
  PY_CONF_PATH="$(cygpath -w "$CONF_PATH")"
fi

cd "$REPO_ROOT"

run_checked uv run python scripts/build_piece_a_p4_drift_trigger_conf.py \
  --phase "$PHASE" \
  --epochs "$EPOCHS" \
  --output "$PY_CONF_PATH"

if [[ "$DRY_RUN" == "true" ]]; then
  cat "$CONF_PATH"
  exit 0
fi

cd "$DEPLOY_DIR"

if [[ "$SKIP_CHECKS" != "true" ]]; then
  run_checked docker compose --env-file ../.env ps airflow-webserver airflow-scheduler iqa-api minio mlflow
  run_checked docker image ls iqa-data
fi

run_checked docker compose --env-file ../.env exec -T airflow-webserver airflow dags unpause iqa_lifecycle_trigger
if [[ "$PHASE" == "confirmed" ]]; then
  run_checked docker compose --env-file ../.env exec -T airflow-webserver airflow dags unpause iqa_lifecycle
else
  run_checked docker compose --env-file ../.env exec -T airflow-webserver airflow dags pause iqa_lifecycle
fi

cat "$CONF_PATH" | docker compose --env-file ../.env exec -T airflow-webserver sh -c \
  'cat > /tmp/iqa_lifecycle_trigger_piece_a_p4_conf.json'

python_trigger="import json, subprocess; conf=json.dumps(json.load(open('/tmp/iqa_lifecycle_trigger_piece_a_p4_conf.json', encoding='utf-8-sig')), separators=(',',':')); subprocess.check_call(['airflow','dags','trigger','iqa_lifecycle_trigger','--conf',conf])"
run_checked docker compose --env-file ../.env exec -T airflow-webserver python -c "$python_trigger"

run_checked docker compose --env-file ../.env exec -T airflow-webserver airflow dags list-runs -d iqa_lifecycle_trigger
run_checked docker compose --env-file ../.env exec -T airflow-webserver airflow dags list-runs -d iqa_lifecycle

if [[ "$PHASE" != "confirmed" && "$KEEP_TRIGGER_UNPAUSED" != "true" ]]; then
  run_checked docker compose --env-file ../.env exec -T airflow-webserver airflow dags pause iqa_lifecycle_trigger
fi
