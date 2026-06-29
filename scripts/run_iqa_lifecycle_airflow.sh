#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_iqa_lifecycle_airflow.sh [options]

Options:
  --scenario-id VALUE                         Default: production_replay_natural_piece_b_full
  --mode VALUE                                Default: progressive-train
  --max-events VALUE                          Default: 372
  --lifecycle-interval VALUE                  Default: 50
  --epochs VALUE                              Default: 8
  --gate-eval-profile VALUE                   Default: full
  --candidate-init-policy fresh|stable_base|active
                                               Default: fresh
  --anchor-good-max-per-class VALUE           Default: 256
  --max-good-red-regression VALUE             Default: 1
  --image-root VALUE                          Default: /opt/iqa/iqa-mlops/.cache/iqa/source_datasets/hss-iad
  --anchor-good-manifest VALUE                Default: data/metadata/feature_ae_bootstrap_piece_b_minimal_v001.csv
  --reference-eval-manifest VALUE             Default: data/validation/validation_set_piece_b_full_v001.csv
  --reference-gt-masks-manifest VALUE         Default: data/validation/validation_gt_masks_piece_b_full_v001.csv
  --ml-image VALUE                            Default: iqa-ml:local
  --target-stage VALUE                        Default: test
  --no-dual-promotion
  --no-classification-require-fn-improvement
  --skip-checks
  --dry-run
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

dotenv_get() {
  local key="$1"
  local line
  line="$(grep -E "^${key}=" "$ENV_FILE" | tail -n 1 || true)"
  printf '%s' "${line#*=}"
}

require_env() {
  local name value
  for name in "$@"; do
    value="$(dotenv_get "$name")"
    if [[ -z "$value" ]]; then
      printf 'Missing required .env variable for Airflow task containers: %s\n' "$name" >&2
      exit 1
    fi
  done
}

assert_airflow_container_env() {
  local source_path target_path
  source_path="$(
    docker compose --env-file ../.env -f docker-compose.yml exec -T airflow-scheduler \
      sh -lc 'printf "%s" "$IQA_AIRFLOW_REPO_MOUNT_SOURCE"'
  )"
  target_path="$(
    docker compose --env-file ../.env -f docker-compose.yml exec -T airflow-scheduler \
      sh -lc 'printf "%s" "$IQA_AIRFLOW_REPO_MOUNT_TARGET"'
  )"

  if [[ "$source_path" != /* || "$target_path" != /* || "$source_path" == *"Program Files/Git"* || "$target_path" == *"Program Files/Git"* ]]; then
    cat >&2 <<EOF
Invalid Airflow container mount env:
  IQA_AIRFLOW_REPO_MOUNT_SOURCE=$source_path
  IQA_AIRFLOW_REPO_MOUNT_TARGET=$target_path

Airflow was likely started from Git Bash with MSYS path conversion enabled.
Restart Airflow from Git Bash with:

  cd /d/MLOPS/deploy
  MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' MSYS2_ENV_CONV_EXCL='*' \\
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml --env-file ../.env up -d --force-recreate \\
    airflow-webserver airflow-scheduler

Then rerun this script.
EOF
    exit 1
  fi
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DEPLOY_DIR="$REPO_ROOT/deploy"
ENV_FILE="$REPO_ROOT/.env"
CACHE_DIR="$REPO_ROOT/.cache/iqa"
CONF_PATH="$CACHE_DIR/iqa_lifecycle_piece_b_full_conf.json"

case "${OSTYPE:-}" in
  msys*|cygwin*)
    export MSYS_NO_PATHCONV=1
    export MSYS2_ARG_CONV_EXCL='*'
    export MSYS2_ENV_CONV_EXCL='*'
    ;;
esac

SCENARIO_ID="production_replay_natural_piece_b_full"
MODE="progressive-train"
MAX_EVENTS="372"
LIFECYCLE_INTERVAL="50"
EPOCHS="8"
GATE_EVAL_PROFILE="full"
CANDIDATE_INIT_POLICY="fresh"
ANCHOR_GOOD_MAX_PER_CLASS="256"
MAX_GOOD_RED_REGRESSION="1"
IMAGE_ROOT="/opt/iqa/iqa-mlops/.cache/iqa/source_datasets/hss-iad"
ANCHOR_GOOD_MANIFEST="data/metadata/feature_ae_bootstrap_piece_b_minimal_v001.csv"
REFERENCE_EVAL_MANIFEST="data/validation/validation_set_piece_b_full_v001.csv"
REFERENCE_GT_MASKS_MANIFEST="data/validation/validation_gt_masks_piece_b_full_v001.csv"
ML_IMAGE="iqa-ml:local"
TARGET_STAGE="test"
DUAL_PROMOTION="true"
CLASSIFICATION_REQUIRE_FN_IMPROVEMENT="true"
SKIP_CHECKS="false"
DRY_RUN="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scenario-id) SCENARIO_ID="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --max-events) MAX_EVENTS="$2"; shift 2 ;;
    --lifecycle-interval) LIFECYCLE_INTERVAL="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --gate-eval-profile) GATE_EVAL_PROFILE="$2"; shift 2 ;;
    --candidate-init-policy) CANDIDATE_INIT_POLICY="$2"; shift 2 ;;
    --anchor-good-max-per-class) ANCHOR_GOOD_MAX_PER_CLASS="$2"; shift 2 ;;
    --max-good-red-regression) MAX_GOOD_RED_REGRESSION="$2"; shift 2 ;;
    --image-root) IMAGE_ROOT="$2"; shift 2 ;;
    --anchor-good-manifest) ANCHOR_GOOD_MANIFEST="$2"; shift 2 ;;
    --reference-eval-manifest) REFERENCE_EVAL_MANIFEST="$2"; shift 2 ;;
    --reference-gt-masks-manifest) REFERENCE_GT_MASKS_MANIFEST="$2"; shift 2 ;;
    --ml-image) ML_IMAGE="$2"; shift 2 ;;
    --target-stage) TARGET_STAGE="$2"; shift 2 ;;
    --no-dual-promotion) DUAL_PROMOTION="false"; shift ;;
    --no-classification-require-fn-improvement) CLASSIFICATION_REQUIRE_FN_IMPROVEMENT="false"; shift ;;
    --skip-checks) SKIP_CHECKS="true"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$CANDIDATE_INIT_POLICY" in
  fresh|stable_base|active) ;;
  *)
    printf 'Invalid --candidate-init-policy: %s (expected fresh, stable_base, or active)\n' "$CANDIDATE_INIT_POLICY" >&2
    exit 2
    ;;
esac

if [[ ! -f "$ENV_FILE" ]]; then
  printf 'Missing env file: %s\n' "$ENV_FILE" >&2
  exit 1
fi

mkdir -p "$CACHE_DIR"

require_env \
  IQA_AIRFLOW_REPO_MOUNT_SOURCE \
  IQA_AIRFLOW_REPO_MOUNT_TARGET \
  IQA_API_URL \
  IQA_SERVICE_TOKEN \
  AWS_ACCESS_KEY_ID \
  AWS_SECRET_ACCESS_KEY \
  IQA_S3_ACCESS_KEY_ID \
  IQA_S3_SECRET_ACCESS_KEY \
  MINIO_ROOT_USER \
  MINIO_ROOT_PASSWORD

export SCENARIO_ID MODE MAX_EVENTS LIFECYCLE_INTERVAL EPOCHS GATE_EVAL_PROFILE
export CANDIDATE_INIT_POLICY ANCHOR_GOOD_MAX_PER_CLASS MAX_GOOD_RED_REGRESSION
export IMAGE_ROOT ANCHOR_GOOD_MANIFEST REFERENCE_EVAL_MANIFEST REFERENCE_GT_MASKS_MANIFEST
export ML_IMAGE TARGET_STAGE DUAL_PROMOTION CLASSIFICATION_REQUIRE_FN_IMPROVEMENT

python_conf_path="$CONF_PATH"
if command -v cygpath >/dev/null 2>&1; then
  python_conf_path="$(cygpath -w "$CONF_PATH")"
fi

"${PYTHON:-python}" - "$python_conf_path" <<'PY'
import json
import os
import sys

conf = {
    "scenario_id": os.environ["SCENARIO_ID"],
    "ml_image": os.environ["ML_IMAGE"],
    "repo_root": "/opt/iqa/iqa-mlops",
    "image_root": os.environ["IMAGE_ROOT"],
    "mode": os.environ["MODE"],
    "max_events": int(os.environ["MAX_EVENTS"]),
    "lifecycle_interval": int(os.environ["LIFECYCLE_INTERVAL"]),
    "max_cycles": None,
    "epochs": int(os.environ["EPOCHS"]),
    "max_steps": None,
    "gate_eval_profile": os.environ["GATE_EVAL_PROFILE"],
    "target_stage": os.environ["TARGET_STAGE"],
    "promotion_min_delta": 0.0,
    "dual_promotion": os.environ["DUAL_PROMOTION"].lower() == "true",
    "localization_promotion_min_delta": 0.0,
    "classification_require_fn_improvement": os.environ["CLASSIFICATION_REQUIRE_FN_IMPROVEMENT"].lower() == "true",
    "classification_min_image_recall_delta": 0.0,
    "classification_min_image_ap_delta": 0.0,
    "anchor_good_manifest": os.environ["ANCHOR_GOOD_MANIFEST"],
    "anchor_good_max_per_class": int(os.environ["ANCHOR_GOOD_MAX_PER_CLASS"]),
    "reference_eval_manifest": os.environ["REFERENCE_EVAL_MANIFEST"],
    "reference_gt_masks_manifest": os.environ["REFERENCE_GT_MASKS_MANIFEST"],
    "max_good_red_regression": int(os.environ["MAX_GOOD_RED_REGRESSION"]),
    "candidate_init_policy": os.environ["CANDIDATE_INIT_POLICY"],
    "require_mlflow_registry": True,
    "mlflow_tracking_uri": "http://mlflow:5000",
    "mlflow_s3_endpoint_url": "http://minio:9000",
    "s3_endpoint_url": "http://minio:9000",
}

with open(sys.argv[1], "w", encoding="utf-8") as file:
    json.dump(conf, file, separators=(",", ":"))
    file.write("\n")
PY

printf 'Config written: %s\n' "$CONF_PATH"

if [[ "$DRY_RUN" == "true" ]]; then
  cat "$CONF_PATH"
  exit 0
fi

pushd "$DEPLOY_DIR" >/dev/null
trap 'popd >/dev/null' EXIT

if [[ "$SKIP_CHECKS" != "true" ]]; then
  run_checked docker compose --env-file ../.env -f docker-compose.yml ps postgres minio mlflow airflow-webserver airflow-scheduler
  assert_airflow_container_env
  run_checked docker compose --env-file ../.env -f docker-compose.yml exec -T minio \
    mc alias set local http://minio:9000 "$(dotenv_get MINIO_ROOT_USER)" "$(dotenv_get MINIO_ROOT_PASSWORD)"
  run_checked docker compose --env-file ../.env -f docker-compose.yml exec -T minio \
    mc ls local/iqa-models/roi_segmenter_v001_fixed/
  run_checked docker compose --env-file ../.env -f docker-compose.yml exec -T minio \
    mc ls local/iqa-models/rd_feature_ae_gated_v001_bootstrap/
  run_checked docker run --rm "$ML_IMAGE" python -c \
    "import torch; print(torch.__version__, torch.version.cuda); assert torch.version.cuda is not None, 'ML image is CPU-only; rebuild with deploy/docker-compose.gpu.yml'"
  run_checked docker run --rm --gpus all "$ML_IMAGE" python -c \
    "import torch; print('CUDA available:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA is not available inside the ML task container'"
  run_checked docker run --rm -v "$(dotenv_get IQA_AIRFLOW_REPO_MOUNT_SOURCE):/probe" "$ML_IMAGE" sh -lc \
    'test -f /probe/data/metadata/casting_flux_replay_plan_piece_b_full_v001.csv && test -d /probe/.cache/iqa/source_datasets/hss-iad && echo MOUNT_OK'
fi

run_checked docker compose --env-file ../.env -f docker-compose.yml exec -T airflow-webserver \
  airflow dags unpause iqa_lifecycle
run_checked docker compose --env-file ../.env -f docker-compose.yml exec -T airflow-webserver \
  airflow pools set iqa_gpu 1 "Single local GPU"

airflow_web="$(docker compose --env-file ../.env -f docker-compose.yml ps -q airflow-webserver)"
if [[ -z "$airflow_web" ]]; then
  printf 'airflow-webserver container was not found.\n' >&2
  exit 1
fi

docker_conf_path="$CONF_PATH"
if command -v cygpath >/dev/null 2>&1; then
  docker_conf_path="$(cygpath -w "$CONF_PATH")"
fi
run_checked docker cp "$docker_conf_path" "${airflow_web}:/tmp/iqa_lifecycle_conf.json"

python_trigger="import json, subprocess; conf=json.dumps(json.load(open('/tmp/iqa_lifecycle_conf.json', encoding='utf-8-sig')), separators=(',',':')); subprocess.check_call(['airflow','dags','trigger','iqa_lifecycle','--conf',conf])"
run_checked docker compose --env-file ../.env -f docker-compose.yml exec -T airflow-webserver \
  python -c "$python_trigger"
run_checked docker compose --env-file ../.env -f docker-compose.yml exec -T airflow-webserver \
  airflow dags list-runs -d iqa_lifecycle
