# 03 - Image ml (inference, trainer)

Type : AFK

## What to build

Produire l'image issue du stage `ml` (torch + torchvision + scikit-learn + mlflow)
et la cabler pour `iqa-inference` et `iqa-trainer` dans le compose. Conserver la
gestion du lock GPU (volume `gpu_lock`).

## Acceptance criteria

- [x] Image `ml` construite depuis le stage multi-stage (`--target ml`, torch 2.12.0+cpu, ~2.23 GB)
- [x] `iqa-inference` demarre et repond sur `:8100` (`GET /health` -> 200 `{"status":"ok","service":"iqa-inference"}`)
- [x] `iqa-trainer` (commande `iqa-run-lifecycle`) s'execute avec torch disponible (console script resolu, torch/torchvision/sklearn/mlflow importes)
- [x] Le lock GPU reste fonctionnel (volume `gpu_lock` monte sur `IQA_GPU_LOCK_PATH`)
- [x] Smoke-test inference vert

## Blocked by

- 02 - Dockerfile multi-stage + image iqa-api slim
