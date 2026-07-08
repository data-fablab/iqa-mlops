# Scenario Drift Piece A/P4

Objectif : demontrer un drift domaine credible en regime etabli, puis une
correction courte et ciblee sans relire tout le replay Piece B.

Le scenario technique est :

```text
production_replay_natural_piece_b_to_piece_a_p4_drift
```

## Fil Narratif

1. Le modele actif issu du scenario Piece B observe un flux stable.
2. Piece A/P4 apparait progressivement dans de petites fenetres de 10 pieces.
3. La premiere fenetre mixte montre une nouveaute ROI partielle, sans correction immediate.
4. Une fenetre P4 complete depasse le seuil `roi_mask_novelty_rate >= 0.8`.
5. Airflow declenche le DAG correctif dedie `iqa_drift_correction_lifecycle`.
6. Le DAG correctif consomme le `drift_context.json`, entraine quelques epochs,
   execute la gate, puis promeut classification et localisation si les criteres
   P4 sont meilleurs que l'actif.

Le drift n'est plus prouve par `domain_ratio`. Le signal metier principal est la
distance des masques ROI au referentiel Piece B :

- `iqa_drift_roi_mask_novelty_rate`
- `iqa_drift_roi_mask_nn_distance`
- `iqa_drift_context_events_total`
- `iqa_drift_status`
- `iqa_drift_trigger_lifecycle`

`scenario_phase` reste une trace de replay, pas une preuve de decision.

## DAGs

### `iqa_drift_piece_a_p4`

Le DAG principal lance `iqa-run-drift-observation-replay`.

Parametres par defaut :

```text
max_events=40
window_size=10
stable_reference_events=10
drift_observation_windows=3
```

La selection courte est construite ainsi :

- 10 evenements Piece B stables pour le referentiel ROI ;
- 1 fenetre mixte avec 2 evenements P4 et 8 evenements Piece B/P1/P2/P3 ;
- 2 fenetres P4 completes pour confirmer et garder un contexte correctif.

Le script produit notamment :

- un resume de run drift ;
- les metriques Prometheus drift ;
- un `drift_context.json` ;
- les manifests cibles utilises par la correction.

Si `trigger_lifecycle=true`, le DAG construit le `conf` de correction et lance
`iqa_drift_correction_lifecycle`.

### `iqa_drift_correction_lifecycle`

Ce DAG est dedie a la correction drift. Il lance
`iqa-run-replay-lifecycle-cycle` avec :

```text
--drift-context-path <drift_context.json>
--max-cycles 1
--epochs 4
--gate-eval-profile fast
--dual-promotion
--skip-report-only-reference-eval
```

Il ne relit pas les 372 evenements Piece B du scenario 1. Le contexte drift et
les manifests equilibres produits par l'observation definissent le perimetre de
correction.

## Donnees

Plan source :

```text
data/metadata/casting_flux_replay_plan_piece_b_to_piece_a_p4_drift_v001.csv
```

Manifests de validation/correction :

```text
data/validation/validation_set_piece_b_to_piece_a_p4_drift_v001.csv
data/validation/classification_selection_piece_b_to_piece_a_p4_drift_v001.csv
data/validation/validation_gt_masks_piece_b_to_piece_a_p4_drift_v001.csv
```

Le scenario conserve Piece B comme reference, puis introduit Piece A/P4 comme
domaine hors referentiel ROI.

## Commandes Airflow

Depuis `deploy/` :

```bash
docker compose --env-file ../.env exec -T airflow-webserver airflow dags unpause iqa_drift_piece_a_p4
docker compose --env-file ../.env exec -T airflow-webserver airflow dags unpause iqa_drift_correction_lifecycle
docker compose --env-file ../.env exec -T airflow-webserver airflow dags trigger iqa_drift_piece_a_p4
```

Observation :

```bash
docker compose --env-file ../.env exec -T airflow-webserver airflow dags list-runs -d iqa_drift_piece_a_p4
docker compose --env-file ../.env exec -T airflow-webserver airflow dags list-runs -d iqa_drift_correction_lifecycle
```

Prometheus/API :

```bash
curl.exe -s http://localhost:8000/metrics | Select-String "iqa_drift"
curl.exe -s http://localhost:8000/metrics | Select-String "iqa_lifecycle"
```

## Dashboard

Le dashboard provisionne `IQA - Drift P4` (`iqa-drift-p4`) raconte le scenario en
bandes horizontales :

1. chronologie drift -> DAG correctif -> promotions ;
2. preuve ROI Piece B stable puis P4 ;
3. contexte cible et train set correctif ;
4. phase DAG correctif : entrainement, gate, promotion ;
5. gates classification/localisation et registry final.

Le dashboard ne doit pas exposer de chemins locaux, masques, images industrielles
ou fichiers `.cache`.

## Garde-fous

- La CI ne declenche jamais ce scenario.
- `iqa_drift_correction_lifecycle` exige un `drift_context_path` pour etre
  operationnellement credible.
- Les metriques API sont reinitialisees par run pour eviter que Grafana affiche
  des valeurs residuelles d'un ancien scenario.
- MLflow Registry reste la source de verite des modeles promus ; MinIO stocke
  les checkpoints et artefacts.
