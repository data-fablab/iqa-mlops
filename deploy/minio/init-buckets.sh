#!/bin/sh
# Creates the IQA buckets in the local MinIO instance.
# Run by the minio-init service (mc client) after minio is up.
set -eu

MINIO_ROOT_USER="${MINIO_ROOT_USER:-minioadmin}"
MINIO_ROOT_PASSWORD="${MINIO_ROOT_PASSWORD:-minioadmin}"

mc alias set iqa-minio http://minio:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD"

for bucket in \
    iqa-source-datasets \
    iqa-dvc \
    iqa-ingested-images \
    mlflow-artifacts \
    iqa-roi-masks \
    iqa-heatmaps \
    iqa-models \
    iqa-backups
do
    mc mb --ignore-existing "iqa-minio/${bucket}"
done
