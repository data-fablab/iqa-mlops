# 18 - Persistance runtime de l'ingestion (events PG / images MinIO)

Type : AFK

## What to build

Implementer la persistance reelle de l'ingestion, aujourd'hui absente :
`scripts/run_ingestion.py` est une frontiere "validated-summary" qui lit le manifeste
CSV et imprime un resume JSON, **sans rien ecrire**. L'issue 07 a conteneurise le DAG ;
celle-ci comble le data plane.

Respecter le contrat lineage (`issues/README.md` > Contrats transverses) :
- **Entree via MinIO**, pas un chemin hote. Le manifeste / les images source sont lus
  depuis MinIO (l'image `data` ne doit pas dependre d'un bind-mount `data/`).
- **Sortie** : events ecrits en PostgreSQL (schema metadata), images materialisees en
  MinIO. XCom ne transporte que des references (URI MinIO, `event_id`).

## Acceptance criteria

- [ ] `iqa-run-ingestion` lit son manifeste/images depuis MinIO (plus de chemin hote)
- [ ] Les events sont persistes en PostgreSQL (table metadata adequate)
- [ ] Les images ingerees sont materialisees en MinIO sous une cle deterministe
- [ ] Idempotence : re-rejouer un meme `scenario_id` ne duplique pas les events
- [ ] Un run du DAG `iqa_ingestion` (issue 07) ecrit reellement dans PG + MinIO
- [ ] Tests : couverture de la couche de persistance ; suite + DagBag verts

## Blocked by

- 07 - DAG ingestion en conteneur

## Note

Meme nature de travail que les criteres de persistance embarques dans les issues
08-13 (cf. cadrage `issues/README.md`). Si l'effort est gros, redecouper
(lecture MinIO / ecriture PG / materialisation images).
