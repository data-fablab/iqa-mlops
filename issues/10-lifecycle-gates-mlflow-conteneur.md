# 10 - Lifecycle (3/4) : gates + enregistrement MLflow en conteneurs

Type : AFK

## What to build

Reecrire `gates` (verification des seuils de promotion) et `mlflow`
(enregistrement du modele dans le MLflow Registry) en taches conteneur. Respecter
l'isolation des modeles par scenario (ADR 0006) :
`feature_ae__production_replay_natural` vs `feature_ae__drift_domain_extension`.

## Acceptance criteria

- [x] `gates` evalue les `configs/promotion_gates.yaml` en conteneur et bloque si echec
      (`iqa-run-gates` sur l'image data : `evaluate_promotion_gates` + exit non-zero
      si une gate echoue). **Reel.** Les metriques candidates arrivent en args (le flux
      depuis un `eval` reel via XCom est cable avec le runtime train/eval, issue 20).
- [~] `mlflow` enregistre le modele dans le Registry avec le nom isole par scenario.
      Le **nom isole** (`feature_ae__<scenario_id>`) est reel ; l'**enregistrement**
      reel est **differe** : `iqa-run-mlflow` est une frontiere "validated-summary"
      (`registered: false`). Isole dans l'**issue 21** (necessite un run_id reel, 20).
- [x] Echec de gate stoppe le DAG avant enregistrement (exit non-zero de `gates`
      -> tache KO -> `mlflow`/`promotion` en aval ne s'executent pas).
- [x] Import DagBag vert (6 taches conteneur + 2 PythonOperator ; garde
      `dag=None` sans provider Docker -> tests `docker_contract` skip).

## Blocked by

- 09 - Lifecycle (2/4) : train + eval

## Suivi

- Enregistrement MLflow reel : **issue 21** (debloquee par celle-ci).
- Conteneurisation du reste du lifecycle : issue 11 (promotion/reload).
