# 11 - Lifecycle (4/4) : promotion + reload en conteneurs

Type : AFK

## What to build

Derniere tranche du lifecycle : `promotion` (passage du modele au stage cible dans
le Registry, ADR 0006 comme source de verite) et `reload` (rechargement du modele
par le service `iqa-inference` via contrat, sans import du runtime). Ferme la boucle
MLOps automatisee.

## Acceptance criteria

- [~] `promotion` promeut le modele au `target_stage` dans MLflow
  (conversion DAG faite : `iqa-run-promotion` sur image ml resout le nom isole +
  la transition `candidate -> target_stage` ; la transition reelle au Registry est
  differee a l'issue 22, `promoted: false`)
- [~] `reload` declenche le rechargement de `iqa-inference` via son contrat HTTP
  (regle de skip reelle : reload seulement si `target_stage == prod` ; l'appel HTTP
  reel est differe a l'issue 22, `reloaded: false`)
- [~] Apres un run complet, `iqa-inference` sert la nouvelle version (issue 22 : runtime)
- [x] `iqa_lifecycle` ne contient plus aucun `PythonOperator`/placeholder (ADR 0008 entierement resolu)
- [x] Import DagBag vert (tests `docker_contract` : 8 task_ids + chaine lineaire intacts)

## Blocked by

- 10 - Lifecycle (3/4) : gates + MLflow

## Sibling (runtime)

- 22 - Runtime promotion + reload : transition Registry reelle + reload HTTP inference
