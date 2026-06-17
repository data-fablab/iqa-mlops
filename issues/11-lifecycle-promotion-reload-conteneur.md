# 11 - Lifecycle (4/4) : promotion + reload en conteneurs

Type : AFK

## What to build

Derniere tranche du lifecycle : `promotion` (passage du modele au stage cible dans
le Registry, ADR 0006 comme source de verite) et `reload` (rechargement du modele
par le service `iqa-inference` via contrat, sans import du runtime). Ferme la boucle
MLOps automatisee.

## Acceptance criteria

- [ ] `promotion` promeut le modele au `target_stage` dans MLflow
- [ ] `reload` declenche le rechargement de `iqa-inference` via son contrat HTTP
- [ ] Apres un run complet, `iqa-inference` sert la nouvelle version
- [ ] `iqa_lifecycle` ne contient plus aucun `PythonOperator`/placeholder (ADR 0008 entierement resolu)
- [ ] Import DagBag vert

## Blocked by

- 10 - Lifecycle (3/4) : gates + MLflow
