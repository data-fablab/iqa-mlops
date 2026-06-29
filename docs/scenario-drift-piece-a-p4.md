# Scenario Drift Piece A/P4

Objectif : valider le flux MLOps naturel sans demarrer un entrainement trop tot.

Le scenario naturel est volontairement separe en trois etapes :

1. Inference-only sur Piece B stable avec les modeles promus du premier scenario.
2. Inference-only avec apparition progressive Piece A/P4.
3. Correction `iqa_lifecycle` declenchee une seule fois si les metriques observees confirment le drift.

Le DAG principal est `iqa_drift_piece_a_p4`. Il lance `iqa-run-drift-observation-replay`, pousse les fenetres sanitisées vers l'API, puis declenche `iqa_lifecycle` uniquement si `trigger_lifecycle=true`.

Le drift n'est pas confirme par la phase du scenario. `domain_ratio` peut rendre le drift suspect, mais la correction exige une degradation observee : alertes, rouges, ROI fail ou faux negatifs oracle.

## Donnees

Plan utilise :

```text
data/metadata/casting_flux_replay_plan_piece_b_to_piece_a_p4_drift_v001.csv
```

Contrat du plan :

```text
stable_baseline_piece_b      372 events
drift_piece_a_p4_suspected    30 events
drift_piece_a_p4_confirmed    25 events
correction_replay             30 events
```

Les events P4 sont issus de `Casting_class1` et utilisent uniquement :

```text
Casting_class1:2_3
```

## Commande Principale

```bash
cd /d/MLOPS/deploy
docker compose --env-file ../.env exec -T airflow-webserver airflow dags unpause iqa_drift_piece_a_p4
docker compose --env-file ../.env exec -T airflow-webserver airflow dags trigger iqa_drift_piece_a_p4
```

La correction demarre avec :

```text
scenario_id=production_replay_natural_piece_b_to_piece_a_p4_drift
candidate_init_policy=active
external_drift_confirmed=true
max_cycles=1
initial_classification_registered_model=feature_ae_classifier__production_replay_natural_piece_b_full
initial_localization_registered_model=feature_ae_localization__production_replay_natural_piece_b_full
mode=progressive-train
reference_eval_manifest=data/validation/validation_set_piece_b_to_piece_a_p4_drift_v001.csv
```

## Smoke Tests Trigger

Les commandes ci-dessous injectent des metriques synthetiques dans `iqa_lifecycle_trigger`. Elles servent a tester le contrat Airflow/monitoring, pas a valider le scenario naturel.

```bash
bash scripts/run_iqa_piece_a_p4_drift_trigger.sh --phase clear --dry-run
bash scripts/run_iqa_piece_a_p4_drift_trigger.sh --phase suspected --dry-run
bash scripts/run_iqa_piece_a_p4_drift_trigger.sh --phase confirmed --dry-run
```

Validation no-drift :

```bash
bash scripts/run_iqa_piece_a_p4_drift_trigger.sh --phase clear
```

Validation drift suspecte, sans correction :

```bash
bash scripts/run_iqa_piece_a_p4_drift_trigger.sh --phase suspected
```

Validation drift confirmee avec declenchement de correction :

```bash
bash scripts/run_iqa_piece_a_p4_drift_trigger.sh \
  --phase confirmed \
  --allow-correction-trigger
```

## Observation

Airflow :

```bash
cd /d/MLOPS/deploy
docker compose --env-file ../.env exec -T airflow-webserver airflow dags list-runs -d iqa_drift_piece_a_p4
docker compose --env-file ../.env exec -T airflow-webserver airflow dags list-runs -d iqa_lifecycle
```

Prometheus/Grafana via API :

```bash
curl.exe -s http://localhost:8002/metrics | Select-String "iqa_drift"
```

## Garde-fou

La phase `confirmed` refuse de s'executer sans `--allow-correction-trigger`, car elle est censee lancer un vrai run de correction.
