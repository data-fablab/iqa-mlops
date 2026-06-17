# 19 - Persistance runtime du dataset candidat (materialisation MinIO / PG)

Type : AFK

## What to build

Implementer la materialisation reelle du dataset candidat, aujourd'hui absente :
`scripts/run_dataset.py` (`iqa-run-dataset`, issue 08) est une frontiere
"validated-summary" qui valide le manifeste et imprime un resume JSON
(`materialized: false`), **sans rien ecrire**. L'issue 08 a conteneurise les taches
`lifecycle_decision` + `dataset` du DAG ; celle-ci comble le data plane du dataset.

Meme nature de travail que l'issue 18 (persistance de l'ingestion) ; reutiliser
les briques data plane existantes (`iqa.metadata.postgres`, `iqa.metadata.repository`,
`iqa.storage.uris`) et `iqa.datasets.build_candidate_dataset`.

Respecter le contrat lineage (`issues/README.md` > Contrats transverses) :
- **Entree via MinIO** (manifeste / images source), pas un bind-mount `data/`.
- **Sortie** : dataset candidat materialise en MinIO sous une cle deterministe et
  enregistre en PostgreSQL (schema metadata). XCom ne transporte que des references
  (URI MinIO, `dataset_version`/`manifest_version`).
- La decision (`lifecycle_decision`) gate la construction (skip si non declenchee).

## Acceptance criteria

- [ ] `iqa-run-dataset` lit son manifeste/images depuis MinIO (plus de chemin hote)
- [ ] Le dataset candidat est materialise en MinIO sous une cle deterministe
- [ ] Le dataset est enregistre en PostgreSQL (version + manifeste lisibles par l'aval)
- [ ] Idempotence : reconstruire une meme `candidate_version` ne duplique pas
- [ ] L'etape `train` (issue 09) peut lire le dataset materialise via reference XCom
- [ ] Un run du DAG `iqa_lifecycle` ecrit reellement dataset en MinIO + PG
- [ ] Tests : couverture de la couche de persistance ; suite + DagBag verts

## Blocked by

- 08 - Lifecycle (1/4) : lifecycle_decision + dataset en conteneurs

## Note

Decoupage decide avec l'issue 08 : la conversion DAG (legere) et la persistance
runtime (lourde) sont deux travaux distincts (cf. cadrage `issues/README.md`).
Si l'effort est gros, redecouper (lecture MinIO / ecriture PG / cle deterministe).
