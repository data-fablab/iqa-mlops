# Cycle de vie du modèle Feature-AE

## Vue d'ensemble

```
dataset (build_candidate_dataset)
    ↓
train (train_feature_ae)
    ↓
eval (evaluate_feature_ae_checkpoint)
    ↓
gates (evaluate_promotion_gates)
    ↓
mlflow (register_run_to_model)
    ↓
promotion (promote_model_with_gates)
    ↓
reload (reload_model_in_inference_service)
```

## Étapes du cycle de vie

### 1. Dataset (`task_dataset`)

**Responsabilité :** Construire un dataset candidat pour l'entraînement.

**Entrées :**
- Scenario ID (détermine le plan de replay : `casting_flux_replay_plan_natural.csv` ou `casting_flux_replay_plan_drift.csv`)
- Validation set figé (`validation_set_v001`) — exclu du dataset candidat (ADR 0001)

**Sorties :**
```python
{
    "path": "/path/to/candidate_dataset.parquet",
    "count": 1000,  # nombre de pieces
    "scenario_id": "production_replay_natural"
}
```

**Implémentation :** `src/iqa/datasets/candidate_builder.py` → `build_candidate_dataset()`

---

### 2. Entraînement (`task_train`)

**Responsabilité :** Entraîner le modèle Feature-AE sur le dataset candidat.

**Entrées :**
- Dataset candidat (path, count)

**Sorties :**
```python
{
    "run_id": "abc123def456",
    "checkpoint": "/path/to/checkpoint.pt",
    "trained_on_count": 1000
}
```

**Logging MLflow :**
- Params: model architecture, learning rate, epochs
- Metrics: train loss, val loss
- Artifacts: checkpoint, hyperparams

**Implémentation :** `src/iqa/training/feature_ae.py` → `train_feature_ae()`

---

### 3. Évaluation (`task_eval`)

**Responsabilité :** Évaluer le modèle entraîné sur le validation set figé.

**Entrées :**
- Checkpoint (du training)
- Validation set v001 (figé, ADR 0001)

**Sorties :**
```python
{
    "recall": 1.0,
    "ap": 0.87,
    "latency_ms": 850,
    "orange_rate": 0.08,
}
```

**Notes :**
- Les metrics sont calculées UNIQUEMENT sur `validation_set_v001`
- Évalue la couverture de défauts par `source_class` (gate `defect_coverage >= 0.95`)

**Implémentation :** `src/iqa/training/feature_ae_evaluation.py` → `evaluate_feature_ae_checkpoint()`

---

### 4. Gates de promotion (`task_gates`)

**Responsabilité :** Vérifier que le candidat passe tous les gates avant enregistrement.

**Entrées :**
- Metrics du candidat (recall, AP, latency, orange_rate)
- Config gates (`configs/promotion_gates.yaml`)

**Sorties :**
```python
{
    "all_passed": True,
    "gates": {
        "recall": {"passed": True, "value": 1.0},
        "ap_regression": {"passed": True, "value": 0.02},
        ...
    },
    "rollback_signal": False
}
```

**Comportement :**
- Si un gate échoue → `Exception` et blocage du pipeline
- Voir `docs/gates.md` pour les seuils détaillés

**Implémentation :** `src/iqa/promotion/__init__.py` → `evaluate_promotion_gates()`

---

### 5. Enregistrement MLflow (`task_mlflow`)

**Responsabilité :** Enregistrer le run entraîné dans MLflow Registry comme `candidate`.

**Entrées :**
- run_id (du training)
- scenario_id (détermine le registered model : `feature_ae__production_replay_natural`, etc.)

**Sorties :**
```python
{
    "registered_model_name": "feature_ae__production_replay_natural",
    "version": "3",
    "stage": "candidate"
}
```

**Logique :**
1. Récupère le run MLflow via `run_id`
2. Crée ou récupère le registered model `feature_ae__{scenario_id}`
3. Crée une nouvelle version avec les artefacts du run
4. Assigne la version au stage `candidate`

**Implémentation :** `src/iqa/registry/mlflow_registry.py` → `register_run_to_model()`

---

### 6. Promotion (`task_promotion`)

**Responsabilité :** Transitionner le candidat du stage `candidate` → `test` (ou `prod` selon config).

**Entrées :**
- registered_model_name
- version
- metrics du candidat

**Sorties :**
```python
{
    "success": True,
    "transition": {"success": True, "from": "candidate", "to": "test"},
    "artifacts": {"artifact_uri": "s3://iqa-models/..."}
}
```

**Logique :**
1. Re-valide les gates (double-check)
2. Transitionne MLflow Registry : `stage: candidate → test`
3. Enregistre la transition dans PostgreSQL (metadata store)

**Implémentation :** `src/iqa/promotion/__init__.py` → `promote_model_with_gates()`

---

### 7. Reload en inférence (`task_reload`)

**Responsabilité :** Charger le nouveau modèle dans le service d'inférence.

**Entrées :**
- scenario_id (détermine le registered model à charger)
- Target stage (par défaut : `prod`)

**Sorties :**
```python
{
    "version": "3",
    "artifact_uri": "s3://iqa-models/feature_ae__production_replay_natural/3/model",
    "registered_model_name": "feature_ae__production_replay_natural"
}
```

**Logique :**
1. Query MLflow Registry : `registered_model_name` + stage `prod`
2. Récupère l'`artifact_uri` (MinIO S3)
3. Charge le modèle dans le service d'inférence via `/admin/reload-model`

**Implémentation :** `src/iqa/inference/model_loader.py` → `ProdModelLoader`

---

## Paramétrage par scénario

### Régimes de replay (`regime`)

Le DAG accepte un paramètre `regime` pour rejouer le cycle selon le scénario :

| Regime | scenario_id | Plan de replay | Cas d'usage |
|--------|-------------|---|---|
| `natural` | `production_replay_natural` | `casting_flux_replay_plan_natural.csv` | Nominal, validation quotidienne |
| `drift` | `drift_domain_extension` | `casting_flux_replay_plan_drift.csv` | Détection drift, stress-test |

**Exemple Airflow :**
```bash
airflow dags trigger iqa_lifecycle \
  --conf '{"regime": "drift", "scenario_id": "drift_domain_extension"}'
```

---

## Archivage et rollback

Voir `docs/rollback.md` pour :
- Transition des versions vers `archived`
- Rollback à une version antérieure
- Gestion des artefacts MinIO lors du rollback

---

## Dépendances externes

| Système | Rôle | Implémentation |
|---------|------|---|
| **Airflow** | Orchestration, DAG, scheduling | `airflow/dags/iqa_lifecycle.py` |
| **MLflow** | Registry (source de vérité), tracking des runs | `src/iqa/registry/mlflow_registry.py` |
| **MinIO** | Stockage des artefacts (S3 compatible) | Buckets: `mlflow-artifacts`, `iqa-models` |
| **PostgreSQL** | Metadata store (transitions, gates, audits) | Configured in `configs/` |
| **Inference Service** | Loading et serving du modèle | `/admin/reload-model` endpoint |

---

## Invariants

1. **Validation set figé** — `validation_set_v001` est exclu de tous les replays (ADR 0001)
2. **Disjonction données** — Aucun chevauchement entre validation, replay, et bootstrap (test automatique)
3. **MLflow source de vérité** — Aucune décision de promotion ne repose sur MinIO (ADR 0006)
4. **Atomicité promotion** — Les transitions MLflow et metadata store PostgreSQL doivent être synchrones
