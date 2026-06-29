param(
    [string]$ScenarioId = "production_replay_natural_piece_b_full",
    [string]$Mode = "progressive-train",
    [int]$MaxEvents = 372,
    [int]$LifecycleInterval = 50,
    [int]$Epochs = 8,
    [string]$GateEvalProfile = "full",
    [ValidateSet("fresh", "stable_base", "active")]
    [string]$CandidateInitPolicy = "fresh",
    [int]$AnchorGoodMaxPerClass = 256,
    [int]$MaxGoodRedRegression = 1,
    [string]$ImageRoot = "/opt/iqa/iqa-mlops/.cache/iqa/source_datasets/hss-iad",
    [string]$AnchorGoodManifest = "data/metadata/feature_ae_bootstrap_piece_b_minimal_v001.csv",
    [string]$ReferenceEvalManifest = "data/validation/validation_set_piece_b_full_v001.csv",
    [string]$ReferenceGtMasksManifest = "data/validation/validation_gt_masks_piece_b_full_v001.csv",
    [string]$MlImage = "iqa-ml:local",
    [string]$TargetStage = "test",
    [switch]$NoDualPromotion,
    [switch]$NoClassificationRequireFnImprovement,
    [switch]$SkipChecks,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )

    Write-Host ">> $Executable $($Arguments -join ' ')"
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Executable $($Arguments -join ' ')"
    }
}

function Read-DotEnv {
    param([Parameter(Mandatory = $true)][string]$Path)

    $values = @{}
    if (-not (Test-Path $Path)) {
        throw "Missing env file: $Path"
    }
    Get-Content -Path $Path | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#")) {
            return
        }
        $index = $line.IndexOf("=")
        if ($index -lt 1) {
            return
        }
        $key = $line.Substring(0, $index).Trim()
        $value = $line.Substring($index + 1).Trim()
        $values[$key] = $value
    }
    return $values
}

function Assert-AirflowContainerEnv {
    $sourcePath = (& docker compose --env-file ../.env -f docker-compose.yml exec -T airflow-scheduler sh -lc 'printf "%s" "$IQA_AIRFLOW_REPO_MOUNT_SOURCE"')
    $targetPath = (& docker compose --env-file ../.env -f docker-compose.yml exec -T airflow-scheduler sh -lc 'printf "%s" "$IQA_AIRFLOW_REPO_MOUNT_TARGET"')

    if (
        -not $sourcePath.StartsWith("/") -or
        -not $targetPath.StartsWith("/") -or
        $sourcePath.Contains("Program Files/Git") -or
        $targetPath.Contains("Program Files/Git")
    ) {
        throw @"
Invalid Airflow container mount env:
  IQA_AIRFLOW_REPO_MOUNT_SOURCE=$sourcePath
  IQA_AIRFLOW_REPO_MOUNT_TARGET=$targetPath

Airflow was likely started from Git Bash with MSYS path conversion enabled.
Restart Airflow from Git Bash with:

  cd /d/MLOPS/deploy
  MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' MSYS2_ENV_CONV_EXCL='*' \
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml --env-file ../.env up -d --force-recreate \
    airflow-webserver airflow-scheduler

Then rerun this script.
"@
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$deployDir = Join-Path $repoRoot "deploy"
$envFile = Join-Path $repoRoot ".env"
$composeFile = Join-Path $deployDir "docker-compose.yml"
$cacheDir = Join-Path $repoRoot ".cache\iqa"
$confPath = Join-Path $cacheDir "iqa_lifecycle_piece_b_full_conf.json"

New-Item -ItemType Directory -Force -Path $cacheDir | Out-Null

$envValues = Read-DotEnv -Path $envFile
$requiredEnv = @(
    "IQA_AIRFLOW_REPO_MOUNT_SOURCE",
    "IQA_AIRFLOW_REPO_MOUNT_TARGET",
    "IQA_API_URL",
    "IQA_SERVICE_TOKEN",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "IQA_S3_ACCESS_KEY_ID",
    "IQA_S3_SECRET_ACCESS_KEY",
    "MINIO_ROOT_USER",
    "MINIO_ROOT_PASSWORD"
)
foreach ($name in $requiredEnv) {
    if (-not $envValues.ContainsKey($name) -or [string]::IsNullOrWhiteSpace($envValues[$name])) {
        throw "Missing required .env variable for Airflow task containers: $name"
    }
}

$conf = [ordered]@{
    scenario_id = $ScenarioId
    ml_image = $MlImage
    repo_root = "/opt/iqa/iqa-mlops"
    image_root = $ImageRoot
    mode = $Mode
    max_events = $MaxEvents
    lifecycle_interval = $LifecycleInterval
    max_cycles = $null
    epochs = $Epochs
    max_steps = $null
    gate_eval_profile = $GateEvalProfile
    target_stage = $TargetStage
    promotion_min_delta = 0.0
    dual_promotion = (-not $NoDualPromotion.IsPresent)
    localization_promotion_min_delta = 0.0
    classification_require_fn_improvement = (-not $NoClassificationRequireFnImprovement.IsPresent)
    classification_min_image_recall_delta = 0.0
    classification_min_image_ap_delta = 0.0
    anchor_good_manifest = $AnchorGoodManifest
    anchor_good_max_per_class = $AnchorGoodMaxPerClass
    reference_eval_manifest = $ReferenceEvalManifest
    reference_gt_masks_manifest = $ReferenceGtMasksManifest
    max_good_red_regression = $MaxGoodRedRegression
    candidate_init_policy = $CandidateInitPolicy
    require_mlflow_registry = $true
    mlflow_tracking_uri = "http://mlflow:5000"
    mlflow_s3_endpoint_url = "http://minio:9000"
    s3_endpoint_url = "http://minio:9000"
}

$conf | ConvertTo-Json -Compress -Depth 10 | Set-Content -Path $confPath -Encoding UTF8
Write-Host "Config written: $confPath"

if ($DryRun) {
    Get-Content -Path $confPath
    exit 0
}

Push-Location $deployDir
try {
    if (-not $SkipChecks) {
        Invoke-Checked docker compose --env-file ../.env -f docker-compose.yml ps postgres minio mlflow airflow-webserver airflow-scheduler
        Assert-AirflowContainerEnv
        Invoke-Checked docker compose --env-file ../.env -f docker-compose.yml exec -T minio mc alias set local http://minio:9000 $envValues["MINIO_ROOT_USER"] $envValues["MINIO_ROOT_PASSWORD"]
        Invoke-Checked docker compose --env-file ../.env -f docker-compose.yml exec -T minio mc ls local/iqa-models/roi_segmenter_v001_fixed/
        Invoke-Checked docker compose --env-file ../.env -f docker-compose.yml exec -T minio mc ls local/iqa-models/rd_feature_ae_gated_v001_bootstrap/
        Invoke-Checked docker run --rm $MlImage python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.version.cuda is not None, 'ML image is CPU-only; rebuild with deploy/docker-compose.gpu.yml'"
        Invoke-Checked docker run --rm --gpus all $MlImage python -c "import torch; print('CUDA available:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA is not available inside the ML task container'"
        Invoke-Checked docker run --rm -v "$($envValues["IQA_AIRFLOW_REPO_MOUNT_SOURCE"]):/probe" $MlImage sh -lc "test -f /probe/data/metadata/casting_flux_replay_plan_piece_b_full_v001.csv && test -d /probe/.cache/iqa/source_datasets/hss-iad && echo MOUNT_OK"
    }

    Invoke-Checked docker compose --env-file ../.env -f docker-compose.yml exec -T airflow-webserver airflow dags unpause iqa_lifecycle
    Invoke-Checked docker compose --env-file ../.env -f docker-compose.yml exec -T airflow-webserver airflow pools set iqa_gpu 1 "Single local GPU"

    $airflowWeb = (& docker compose --env-file ../.env -f docker-compose.yml ps -q airflow-webserver).Trim()
    if (-not $airflowWeb) {
        throw "airflow-webserver container was not found."
    }
    Invoke-Checked docker cp $confPath "${airflowWeb}:/tmp/iqa_lifecycle_conf.json"

    $pythonTrigger = "import json, subprocess; conf=json.dumps(json.load(open('/tmp/iqa_lifecycle_conf.json', encoding='utf-8-sig')), separators=(',',':')); subprocess.check_call(['airflow','dags','trigger','iqa_lifecycle','--conf',conf])"
    Invoke-Checked docker compose --env-file ../.env -f docker-compose.yml exec -T airflow-webserver python -c $pythonTrigger
    Invoke-Checked docker compose --env-file ../.env -f docker-compose.yml exec -T airflow-webserver airflow dags list-runs -d iqa_lifecycle
}
finally {
    Pop-Location
}
