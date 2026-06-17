# 03 - Image ml (inference, trainer)

Type : AFK

## What to build

Produire l'image issue du stage `ml` (torch + torchvision + scikit-learn + mlflow)
et la cabler pour `iqa-inference` et `iqa-trainer` dans le compose. Conserver la
gestion du lock GPU (volume `gpu_lock`).

## Acceptance criteria

- [ ] Image `ml` construite depuis le stage multi-stage
- [ ] `iqa-inference` demarre et repond sur `:8100`
- [ ] `iqa-trainer` (commande `iqa-run-lifecycle`) s'execute avec torch disponible
- [ ] Le lock GPU reste fonctionnel (volume monte, un seul detenteur)
- [ ] Smoke-test inference vert

## Blocked by

- 02 - Dockerfile multi-stage + image iqa-api slim
