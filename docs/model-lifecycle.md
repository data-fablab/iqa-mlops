# Cycle de vie du modele Feature-AE

## Statut

Le DAG `iqa_lifecycle` est un workflow Phase 1 de staging/test controle. Il est
importable dans Airflow Docker et execute des taches coherentes, mais il ne doit
pas etre presente comme une promotion production automatique.

Par defaut :

```text
lifecycle_decision -> dataset -> train -> eval -> gates -> mlflow -> alias test -> reload skipped
```

En mode production explicite :

```text
lifecycle_decision -> dataset -> train -> eval -> gates -> mlflow -> save previous prod -> alias prod -> reload prod
```

Le mode production necessite `target_stage="prod"` dans les parametres Airflow.

## Declencheurs data-event

Le lifecycle Feature-AE est declenche par evenement donnees. La CI ne declenche
jamais un entrainement modele ; elle verifie uniquement le code, les contrats et
la reproductibilite.

Regles Phase 2 operationnelles :

| Scenario | Declencheur | Dataset candidat |
| --- | --- | --- |
| `production_replay_natural` | au moins 50 nouveaux `piece_event` conformes valides par `oracle_gt` | `feature_ae_good_v002` |
| `drift_domain_extension` | `drift_confirmed=true` | `feature_ae_good_v003` |

Ces regles materialisent la decision projet historique : reentrainement apres
volume suffisant de conformes valides, lot complet ou drift confirme. Le seuil
retenu pour le replay naturel est 50, dans la plage documentee de 30 a 50
nouvelles pieces conformes validees.

Le batch monitoring expose une decision structuree (`lifecycle_decision`) avec
la raison du declenchement et le `candidate_dataset_version`. Airflow consomme
cette decision ; le mode par defaut reste `target_stage=test` et ne promeut pas
automatiquement en production.

Le DAG `iqa_lifecycle` consomme cette meme decision en premiere tache. Si
`trigger_lifecycle=false`, les taches aval retournent un statut `skipped` et ne
lancent pas de training. Si `trigger_lifecycle=true`, le `candidate_dataset_version`
alimente le `dataset_version` candidat, et `manifest_version` est propage vers
les parametres et tags MLflow.

## Taches

### Collecte des signaux lifecycle

Le DAG horaire `iqa_lifecycle_trigger` execute
`iqa-collect-lifecycle-signal` dans l'image data.

Le collecteur lit PostgreSQL et construit les signaux suivants :

- nouveaux feedbacks fermes `oracle_gt`, conformes et eligibles ;
- dernier evenement drift versionne non consomme ;
- taux `roi_fail_rate` sur une fenetre configurable.

Chaque decision, positive ou negative, est journalisee dans
`lifecycle_trigger_events`. Les decisions positives enregistrent aussi les
identifiants de predictions ou de drift consommes et leur watermark. Un meme
signal ne peut donc pas declencher deux fois le lifecycle apres un nouveau
polling ou un redemarrage.

Airflow utilise deux branches independantes, naturel et drift. Chaque branche
suit le chemin conteneur de collecte, gate sans training si la decision est
negative, puis `TriggerDagRunOperator` vers `iqa_lifecycle` si elle est positive.
Le DAG impose `max_active_runs=1`.

### `task_dataset`

Construit le dataset candidat depuis le manifest et l'image root. Si
`roi_predictions_dirs` est fourni, les statuts ROI sont charges via
`load_roi_mask_lookup(...)` puis transmis a `build_candidate_dataset(...)`.

Sans index ROI, la tache conserve le comportement MVP et retourne un warning
explicite indiquant que le filtrage ROI n'a pas ete applique.

### `task_train`

Lance l'entrainement Feature-AE avec les metadonnees attendues :

- `candidate_version`
- `dataset_version`
- `roi_model_version` (`roi_segmenter_v001_fixed` par defaut)
- `feature_ae_version` (`rd_feature_ae_gated_v001_bootstrap` par defaut)

Le training loggue le checkpoint sous l'artefact MLflow `model/` et retourne le
`run_id`.

### `task_eval`

Evalue le checkpoint candidat et retourne les metriques disponibles pour les
gates : recall, AP, orange rate et latence. Les metriques de type
`defect_coverage` complet et `roi_fail_rate` complet restent des cibles si elles
ne sont pas calculees dans cette tache.

### `task_gates`

Charge `configs/promotion_gates.yaml` puis appelle
`evaluate_promotion_gates(...)`. Le gate `ap_regression` ne peut etre evalue que
si `prod_ap` est disponible.

### `task_mlflow`

Enregistre le run comme version du registered model `feature_ae__{scenario_id}`.
La version est creee depuis l'artefact checkpoint du run, puis l'alias
`candidate` est positionne.

### `task_promotion`

Par defaut, promeut le candidat vers l'alias `test`.

Si `target_stage="prod"`, la tache sauvegarde d'abord l'ancien alias `prod`, puis
promeut le candidat vers `prod`.

### `task_reload`

Recharge uniquement le modele `prod`. Pour le mode par defaut `test`, la tache
retourne un statut `skipped` avec une raison explicite.

## Parametres Airflow importants

| Parametre | Defaut | Role |
|---|---|---|
| `scenario_id` | `production_replay_natural` | Registered model cible |
| `conforming_validated_count` | `0` | Nombre de conformes valides `oracle_gt` pour le replay naturel |
| `drift_confirmed` | `false` | Declencheur explicite du scenario drift |
| `roi_fail_rate` | `0.0` | Signal monitoring conserve pour audit |
| `target_stage` | `test` | Alias MLflow cible |
| `manifest_version` | derivee de `candidate_version` | Version du manifest candidat tracee dans MLflow |
| `roi_predictions_dirs` | vide | Index de predictions ROI a appliquer au dataset |
| `gates_config_path` | `configs/promotion_gates.yaml` | Config des gates |

## Invariants

- La validation set reste exclue des datasets candidats.
- MLflow Registry reste la source de verite pour les aliases modele.
- MinIO stocke les artefacts mais ne decide pas du modele actif.
- Le reload production n'est autorise que pour `target_stage="prod"`.
