# 04 - Image data (ingestion, replay, monitoring)

Type : AFK

## What to build

Produire l'image issue du stage `data` (pandas, pillow, boto3, psycopg, sans torch)
et la cabler pour `iqa-ingestion`, `iqa-replay`, `iqa-monitoring`.

## Acceptance criteria

- [ ] Image `data` construite depuis le stage multi-stage, sans torch
- [ ] `iqa-run-ingestion`, `iqa-run-replay`, `iqa-run-monitoring` s'executent dans l'image data
- [ ] Acces MinIO/PostgreSQL fonctionnel depuis l'image data
- [ ] Profils `batch` du compose pointent vers l'image data

## Blocked by

- 02 - Dockerfile multi-stage + image iqa-api slim
