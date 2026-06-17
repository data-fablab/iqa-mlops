# 22 - Runtime promotion + reload : transition Registry reelle + reload HTTP inference

Type : AFK

## What to build

Implementer le runtime reel derriere les deux dernieres frontieres du lifecycle,
aujourd'hui en `validated-summary` :

- `scripts/run_promotion.py` (`iqa-run-promotion`, issue 11) resout le nom isole
  par scenario et la transition `candidate -> target_stage`, et signale si une
  sauvegarde du prod courant est requise (`snapshot_previous_prod`), **sans
  transitionner** (`promoted: false`). Reutiliser
  `iqa.promotion.promote_model_with_gates` + `save_previous_prod_before_promotion`
  (deja presents) pour effectuer la transition reelle au Registry (MLflow source
  de verite, ADR 0006).
- `scripts/run_reload.py` (`iqa-run-reload`, issue 11) applique deja la regle
  reelle (reload seulement si `target_stage == prod`) et resout le nom, mais
  **n'appelle pas** le contrat de rechargement de `iqa-inference`
  (`reloaded: false`). Cabler le reload reel via le contrat HTTP du service
  (a exposer cote `iqa-inference` : endpoint de hot-reload du modele prod) en
  s'appuyant sur `iqa.inference.model_loader.ProdModelLoader`.

L'issue 11 a conteneurise les taches `promotion` + `reload` du DAG (ADR 0008
entierement resolu) ; celle-ci comble le runtime.

## Acceptance criteria

- [ ] `iqa-run-promotion` transitionne reellement le modele au `target_stage` dans
  le Registry, et sauvegarde le prod courant avant une promotion prod (rollback)
- [ ] `iqa-run-reload` declenche le rechargement de `iqa-inference` via son contrat
  HTTP (un endpoint de reload existe cote service)
- [ ] Apres un run complet jusqu'a `prod`, `iqa-inference` sert la nouvelle version
- [ ] La regle de skip (`target_stage != prod` => reload inactif) reste vraie
- [ ] Tests : couverture de la transition + du reload ; suite + DagBag verts

## Blocked by

- 11 - Lifecycle (4/4) : promotion + reload
- 21 - Runtime MLflow (fournit une version reellement enregistree a promouvoir)

## Note

Decoupage coherent avec 07->18, 08->19, 09->20, 10->21 : la conversion DAG
(legere ; le nom isole, la regle `snapshot_previous_prod` et le skip non-prod sont
deja reels) et le runtime (transition Registry + reload HTTP) sont deux travaux
distincts (cf. cadrage `issues/README.md`).
