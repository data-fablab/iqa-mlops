# 09 - Lifecycle (2/4) : train + eval en conteneurs (pool GPU)

Type : AFK

## What to build

Reecrire les etapes `train` et `eval` en taches conteneur sur l'image `ml`, avec le
pool `iqa_gpu` (concurrence limitee a un entrainement). Le checkpoint produit est
ecrit en MinIO et consommable par l'evaluation puis les gates.

## Acceptance criteria

- [ ] `train` et `eval` s'executent en conteneurs image `ml`
- [ ] Pool `iqa_gpu` respecte : un seul entrainement concurrent
- [ ] Lock GPU detenu pendant train/eval, libere ensuite
- [ ] Checkpoint et metriques d'eval persistes (MinIO) et accessibles a l'etape gates
- [ ] Import DagBag vert

## Blocked by

- 08 - Lifecycle (1/4) : lifecycle_decision + dataset
- 03 - Image ml (inference, trainer)
