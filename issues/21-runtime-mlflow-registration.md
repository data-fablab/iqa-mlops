# 21 - Runtime MLflow : enregistrement reel du run au Registry

Type : AFK

## What to build

Implementer l'enregistrement reel du modele au MLflow Registry, aujourd'hui absent :
`scripts/run_mlflow.py` (`iqa-run-mlflow`, issue 10) resout le nom isole par scenario
(`feature_ae__<scenario_id>`, reel) et imprime un resume JSON (`registered: false`),
**sans enregistrer**. L'issue 10 a conteneurise les taches `gates` + `mlflow` du
DAG ; celle-ci comble le runtime Registry.

Reutiliser `iqa.registry.mlflow_registry.register_run_to_model` (deja present).

Respecter le contrat lineage (`issues/README.md` > Contrats transverses) :
- **Entree** : `run_id` d'un entrainement reel (issue 20), via reference XCom.
- **Sortie** : version enregistree au Registry sous le nom isole par scenario
  (ADR 0006) ; XCom ne transporte que des references (nom, version, stage).
- MLflow reste la source de verite de la promotion (le Registry decide, pas MinIO).

## Acceptance criteria

- [ ] `iqa-run-mlflow` enregistre reellement le run au Registry (`register_run_to_model`)
- [ ] Le nom enregistre est isole par scenario (`feature_ae__<scenario_id>`)
- [ ] La version + le stage `candidate` sont lisibles par la promotion (issue 11)
- [ ] Idempotence : re-enregistrer un meme run_id ne cree pas de doublon incoherent
- [ ] Tests : couverture de l'enregistrement ; suite + DagBag verts

## Blocked by

- 10 - Lifecycle (3/4) : gates + enregistrement MLflow
- 20 - Runtime train/eval (fournit le run_id reel)

## Note

Decoupage coherent avec 07->18, 08->19, 09->20 : la conversion DAG (legere ; le nom
isole et le blocage des gates sont deja reels) et le runtime Registry (enregistrement
reel) sont deux travaux distincts (cf. cadrage `issues/README.md`). Les `gates`
(issue 10) sont deja reelles et bloquantes ; seul l'enregistrement MLflow reste a
cabler ici.
