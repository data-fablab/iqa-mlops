# 09 - Lifecycle (2/4) : train + eval en conteneurs (pool GPU)

Type : AFK

## What to build

Reecrire les etapes `train` et `eval` en taches conteneur sur l'image `ml`, avec le
pool `iqa_gpu` (concurrence limitee a un entrainement). Le checkpoint produit est
ecrit en MinIO et consommable par l'evaluation puis les gates.

## Acceptance criteria

- [x] `train` et `eval` s'executent en conteneurs image `ml` via `make_container_task`
      (`iqa-run-train`, `iqa-run-eval` ; image `{{ params.ml_image }}`).
- [x] Pool `iqa_gpu` respecte : un seul entrainement concurrent (`pool=GPU_POOL`,
      slots=1).
- [x] Lock GPU detenu pendant train/eval, libere ensuite : `gpu_lock=True` (factory
      monte le volume + `IQA_GPU_LOCK_PATH`) et les scripts acquierent
      `iqa.runtime.gpu_lock(owner=...)` pour toute leur duree (`--wait-for-gpu`).
- [~] Checkpoint et metriques d'eval persistes (MinIO) et accessibles a l'etape gates.
      **Differe** : `iqa-run-train`/`iqa-run-eval` sont des frontieres
      "validated-summary" (`persisted: false`), sans entrainement reel. Le runtime ML
      + materialisation MinIO est isole dans l'**issue 20** (meme decoupage que
      07->18 et 08->19 ; cf. cadrage `issues/README.md`).
- [x] Import DagBag vert (4 taches conteneur + 4 PythonOperator ; garde
      `dag=None` sans provider Docker -> tests `docker_contract` skip).

## Blocked by

- 08 - Lifecycle (1/4) : lifecycle_decision + dataset
- 03 - Image ml (inference, trainer)

## Suivi

- Runtime ML reel (entrainement + checkpoint/metriques MinIO) : **issue 20**
  (debloquee par celle-ci).
- Conteneurisation du reste du lifecycle : issues 10 (gates/mlflow),
  11 (promotion/reload).
