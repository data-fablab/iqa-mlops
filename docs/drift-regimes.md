# Régimes de drift — Naturel vs Drift

## Concept

Un **régime** est un scénario de réplay de données qui teste le modèle sous conditions différentes.

Deux régimes sont activés pour le MVP :

1. **Naturel (`production_replay_natural`)** — Tester le modèle sur des données nominales
2. **Drift (`drift_domain_extension`)** — Tester le modèle sur des données dégradées ou d'extension de domaine

---

## Régime Naturel

### Définition

Rejoue le **cycle nominal** : les données du casting sont représentatives des conditions de production attendues.

**Cas d'usage :**
- Validation quotidienne de la stabilité du modèle
- Rebaseline après mise à jour intentionnelle
- Confiance opérationnelle

### Plan de replay

Fichier : `data/metadata/casting_flux_replay_plan_natural.csv`

```
simulated_event_id,scenario_id,scenario_type,scenario_phase,is_representative,sequence_number,...
sim_event_ac08a67addc5,production_replay_natural,production,natural_replay,true,1,...
sim_event_d699dedfade5,production_replay_natural,production,natural_replay,true,2,...
```

**Caractéristiques :**
- `scenario_id`: `production_replay_natural`
- `scenario_phase`: `natural_replay`
- `is_representative`: `true` (toutes les observations sont représentatives)
- Défauts : distribués selon proportions attendues en production

### Registered Model

**Nom :** `feature_ae__production_replay_natural`

**Versions :**
- v1, v2, v3, ... (une version par entraînement)

**Stages :**
- `candidate` — En évaluation après entraînement
- `staging` — Candidat passé les gates, prêt pour test
- `prod` — Actif en production
- `archived` — Remplacé par une version plus récente

### Seuils de promotion (naturel)

Voir `configs/promotion_gates.yaml` section `feature_ae` :

```yaml
feature_ae:
  recall_defect_min: 1.0  # Aucun faux négatif
  false_negative_total_max: 0
  image_ap_max_regression: 0.02  # Régression max vs prod
  roi_fail_rate_max: 0.10
  latency_p95_ms_max: 1000
  defect_coverage:
    min_coverage: 0.95  # Au moins 95 % des classes représentées
```

---

## Régime Drift

### Définition

Teste le modèle sous **conditions dégradées** ou d'**extension de domaine** :
- Classes visuelles nouvelles (éclairage, angle, contraste)
- Défauts morphologiquement différents
- Distribution décalée vs. production

**Cas d'usage :**
- Détection de drift en production
- Robustesse aux extensions de domaine
- Stress-testing du modèle
- Baseline pour alerte MLOps

### Plan de replay

Fichier : `data/metadata/casting_flux_replay_plan_drift.csv`

```
simulated_event_id,scenario_id,scenario_type,scenario_phase,is_representative,sequence_number,...
sim_event_0519ccea47d9,drift_domain_extension,mlops_stress_test,baseline_domain_class1,false,1,...
sim_event_a115784cb7bd,drift_domain_extension,mlops_stress_test,baseline_domain_class1,false,2,...
```

**Caractéristiques :**
- `scenario_id`: `drift_domain_extension`
- `scenario_type`: `mlops_stress_test`
- `scenario_phase`: `baseline_domain_class1`, `domain_extension_class2`, etc.
- `is_representative`: `false` (données synthétiques ou de distribution décalée)
- Défauts : augmentés, morphologiquement altérés ou hors-distribution

### Registered Model

**Nom :** `feature_ae__drift_domain_extension`

**Versions :**
- v1, v2, v3, ... (une version par entraînement avec données drift)

**Stages :**
- `candidate`, `staging`, `prod`, `archived` (mêmes règles que naturel)

### Seuils de promotion (drift)

**Décision :** Les seuils drift sont **plus relâchés** que ceux du régime naturel.

```yaml
feature_ae_drift:
  recall_defect_min: 0.98  # Tolérance 2 % faux négatifs (vs 0 % naturel)
  false_negative_total_max: 5  # Quelques faux négatifs acceptés
  image_ap_max_regression: 0.10  # Régression plus acceptable
  roi_fail_rate_max: 0.15  # Légèrement plus permissif
  latency_p95_ms_max: 1200  # Latence un peu plus haute acceptable
  defect_coverage:
    min_coverage: 0.90  # 90 % des classes (vs 95 % naturel)
```

**Justification :** Le drift est par définition hors-domaine ; un modèle ne peut pas être aussi performant.

---

## Baseline drift

### Concept

Une **baseline drift** est la première version du modèle entraîné sur données naturelles, **évaluée sur le plan drift**.

**Objectif :**
- Établir une référence de dégradation attendue sous drift
- Détecter si une version nouvelle est significativement MIEUX ou PIRE que la baseline

**Logique :**
```
v1_natural (entraîné sur naturel)
    ↓
évaluée sur validation_set_v001 → metrics_naturel ✓
    ↓
entraîné sur drift
v1_drift ← (même architecture, données différentes)
    ↓
évaluée sur validation_set_v001 → metrics_drift_baseline
```

### Utilisation

Lors de la promotion d'une nouvelle version drift :

```python
candidate_metrics_drift = evaluate(v_new_drift, validation_set_v001)
baseline_metrics_drift = get_mlflow_metrics("feature_ae__drift_domain_extension", stage="prod")

# Gate : régression acceptable ?
regression = candidate_metrics_drift["ap"] - baseline_metrics_drift["ap"]
assert regression < 0.05  # Max 5% régression AP
```

---

## Exécution des régimes

### DAG Airflow avec paramètres

Le DAG `iqa_lifecycle` accepte les paramètres :

```bash
# Régime naturel (défaut)
airflow dags trigger iqa_lifecycle

# Régime drift
airflow dags trigger iqa_lifecycle \
  --conf '{
    "regime": "drift",
    "scenario_id": "drift_domain_extension"
  }'
```

### Context Airflow

Les tâches reçoivent les paramètres via le contexte :

```python
def task_dataset(**context):
    regime = context.get("regime", "natural")
    scenario_id = context.get("scenario_id", "production_replay_natural")

    # Charger le bon plan
    if regime == "natural":
        plan_file = "casting_flux_replay_plan_natural.csv"
    else:
        plan_file = "casting_flux_replay_plan_drift.csv"

    return build_candidate_dataset(plan_file=plan_file)
```

---

## Monitoring et alertes

### Seuils d'alerte (drift)

| Métrique | Naturel | Drift | Alerte si |
|----------|---------|-------|-----------|
| **Recall** | ≥ 1.0 | ≥ 0.98 | < seuil |
| **AP** | ≥ 0.85 | ≥ 0.80 | < seuil |
| **Orange rate** | ≤ 0.05 | ≤ 0.12 | > seuil |
| **Latency p95** | ≤ 1000 ms | ≤ 1200 ms | > seuil |
| **Defect coverage** | ≥ 0.95 | ≥ 0.90 | < seuil |

---

## Évolution future

**Phase 2 :** Ajouter des régimes supplémentaires :
- `seasonal_variation` — données saisonnières
- `tool_wear_simulation` — dégradation des outils
- `camera_calibration_drift` — décalage de caméra
- `environmental_extremes` — conditions extrêmes

Chacun aurait son propre :
- Plan de replay CSV
- Registered model MLflow
- Seuils de gates ajustés
- Monitoring spécifique
