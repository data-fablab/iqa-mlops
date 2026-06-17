# 20 - Runtime train/eval : entrainement reel + checkpoint/metriques MinIO

Type : AFK

## What to build

Implementer l'execution reelle de l'entrainement et de l'evaluation, aujourd'hui
absente des frontieres conteneur : `scripts/run_train.py` (`iqa-run-train`) et
`scripts/run_eval.py` (`iqa-run-eval`, issue 09) acquierent le lock GPU et impriment
un resume JSON (`persisted: false`), **sans entrainer ni evaluer**. L'issue 09 a
conteneurise les taches `train` + `eval` du DAG (image ml, pool `iqa_gpu`, lock GPU) ;
celle-ci comble le runtime ML.

Reutiliser les briques existantes : `scripts/train_feature_ae.py`
(`iqa-train-feature-ae`), `scripts/evaluate_feature_ae.py`, `iqa.training`,
`iqa.training.mlflow_logging`, et les briques data plane (`iqa.storage.uris`).

Respecter le contrat lineage (`issues/README.md` > Contrats transverses) :
- **Entree** : dataset candidat materialise (issue 19) lu via reference XCom.
- **Sortie** : checkpoint materialise en MinIO sous une cle deterministe ;
  metriques d'eval persistees (MinIO/PG) et lisibles par les gates (issue 10).
  XCom ne transporte que des references (URI checkpoint, run_id, version).

## Acceptance criteria

- [ ] `iqa-run-train` entraine reellement (ou delegue a `iqa-train-feature-ae`)
- [ ] Le checkpoint est materialise en MinIO sous une cle deterministe
- [ ] `iqa-run-eval` evalue reellement le checkpoint et produit les metriques
- [ ] Metriques d'eval persistees (MinIO/PG) et accessibles a l'etape gates
- [ ] Lock GPU detenu pendant le travail reel, libere ensuite (deja le cas en 09)
- [ ] Idempotence : re-entrainer une meme version ne corrompt pas l'artefact
- [ ] Tests : couverture train/eval runtime ; suite + DagBag verts

## Blocked by

- 09 - Lifecycle (2/4) : train + eval en conteneurs
- 19 - Persistance runtime du dataset candidat (entree de train)

## Note

Decoupage coherent avec 07->18 et 08->19 : la conversion DAG (legere, lock GPU
verifiable) et le runtime ML reel (lourd, GPU/torch) sont deux travaux distincts
(cf. cadrage `issues/README.md`).
