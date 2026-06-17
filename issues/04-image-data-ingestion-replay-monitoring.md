# 04 - Image data (ingestion, replay, monitoring)

Type : AFK

## What to build

Produire l'image issue du stage `data` (pandas, pillow, boto3, psycopg, sans torch)
et la cabler pour `iqa-ingestion`, `iqa-replay`, `iqa-monitoring`.

## Acceptance criteria

- [x] Image `data` construite depuis le stage multi-stage, sans torch (`--target data`, ~540 MB, `torch` absent)
- [x] `iqa-run-ingestion`, `iqa-run-replay`, `iqa-run-monitoring` s'executent dans l'image data (console scripts resolus, argparse OK)
- [x] Acces MinIO/PostgreSQL : `pandas`/`boto3`/`psycopg` importes ; bout-en-bout couvert par `deploy/smoke-test.sh` sur la stack compose
- [x] Profils `batch` du compose pointent vers l'image data (`target: data` dans `docker-compose.yml`)

## Blocked by

- 02 - Dockerfile multi-stage + image iqa-api slim
