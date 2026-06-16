# MLflow Registry — Source de vérité

## Décision

**MLflow Registry est la source de vérité unique** pour la promotion, le rollback et l'activité des modèles en production.

**MinIO stocke les artefacts** (checkpoints, heatmaps, datasets) mais ne décide pas quel modèle est actif.

Voir :
- **ADR 0006** — MLflow Registry comme source de vérité
- **ADR 0003** — MinIO comme stockage objet local

---

## Architecture de promotion

```
scenario_id (e.g., feature_ae__production_replay_natural)
    ↓
MLflow Registry → registered model
    ↓
stage: prod | staging | dev | archived
    ↓
artifact_uri: s3://mlflow-artifacts/... ou s3://iqa-models/...
    ↓
/admin/reload-model → reload inference service
```

## Registered Models par scénario

Un modèle registered est créé par **scénario de données et algorithme** :

```text
feature_ae__production_replay_natural  ← Replay naturel (nominal)
feature_ae__drift_domain_extension     ← Replay drift (stress-test)
roi__surface_defects
teacher__resnet18_imagenet_frozen
```

### Convention de nommage

**Format :** `<algorithm>__<scenario_id>`

- `algorithm` : base du modèle (feature_ae, roi_segmenter, teacher, etc.)
- `scenario_id` : mode de données (production_replay_natural, drift_domain_extension, etc.)

**Avantages :**
- Chaque scénario a son propre historique de versions
- Métriques comparables : v1 naturel vs v2 naturel (même données)
- Pas d'ambiguïté : v1 drift ≠ v1 naturel

### Versions et Stages

Chacun peut avoir plusieurs versions, avec des **stages** distincts :
- `prod` — modèle actif en production (un seul à la fois par registered model)
- `staging` — candidat approuvé, prêt pour déploiement
- `dev` — expérimentations, candidates pas encore validées
- `archived` — versions remplacées ou invalidées

**Invariant MLflow :** Un seul registered model + version en stage `prod` à la fois.

## Stockage MinIO

Buckets associés à la décision MLflow :

| Bucket | Contenu | Lien à ADR 0003 |
|--------|---------|-----------------|
| `mlflow-artifacts` | Artefacts MLflow (modèles trackés, métriques, params) | stockage direct |
| `iqa-models` | Modèles promus, candidats archivés | S3 compatible |
| `iqa-heatmaps` | Heatmaps de prédictions | S3 compatible |
| `iqa-ingested-images` | Images brutes ingérées/rejouées | S3 compatible |
| `iqa-dvc` | Datasets DVC (endpointUrl MinIO) | S3 compatible |
| `iqa-source-datasets` | Dataset source Casting | S3 compatible |

**Invariant** : MinIO est un **stockage passif**. Aucune logique de promotion ou rollback ne dépend d'un préfixe S3 comme `s3://iqa-models/prod`.

---

## Configuration

### Chemins (configs/paths.yaml)
```yaml
storage:
  models_bucket: "iqa-models"
  mlflow_bucket: "mlflow-artifacts"
  heatmaps_bucket: "iqa-heatmaps"
  ingested_images_bucket: "iqa-ingested-images"
  dvc_bucket: "iqa-dvc"
  source_datasets_bucket: "iqa-source-datasets"
```

### Environnement (.env)
```
IQA_MLFLOW_TRACKING_URI=http://localhost:5000
IQA_MLFLOW_REGISTRY_SOURCE_OF_TRUTH=true
IQA_S3_ENDPOINT_URL=http://localhost:9000
IQA_ACTIVE_MODEL_ALIAS=prod
```

---

## Source de vérité — Résolution du modèle actif

### Flux de décision

Pour charger le modèle actif en inférence :

```
1. User / Service fournit : scenario_id
   Exemple: "production_replay_natural"

2. Construit registered_model_name
   Format: f"feature_ae__{scenario_id}"
   Résultat: "feature_ae__production_replay_natural"

3. Query MLflow Registry
   Récupère: (registered_model_name, stage="prod")
   Retourne: version, artifact_uri

4. Récupère artifact
   S3: artifact_uri = "s3://mlflow-artifacts/.../model/"
   ou  artifact_uri = "s3://iqa-models/.../model/"
   (MinIO est le stockage, pas la source de vérité)

5. Load modèle
   Charge checkpoint depuis S3 URI
   Enregistre métadonnées dans PostgreSQL (audit trail)
```

### Code exemple

```python
def reload_model_for_scenario(scenario_id: str):
    """Récupère le modèle actif en prod pour un scénario."""

    # Étape 1-2: Construit le nom
    registered_model_name = f"feature_ae__{scenario_id}"

    # Étape 3: Query MLflow (source de vérité)
    versions = mlflow.client.get_latest_versions(
        name=registered_model_name,
        stages=["prod"]
    )

    if not versions:
        raise ValueError(f"No prod version for {registered_model_name}")

    prod_version = versions[0]  # Un seul par invariant MLflow

    # Étape 4: Récupère URI depuis MLflow
    artifact_uri = prod_version.source  # e.g., "s3://mlflow-artifacts/.../model"

    # Étape 5: Load depuis MinIO (stockage passif)
    model = load_model_from_s3(artifact_uri)

    return model
```

### Garanties de la source de vérité MLflow

1. **Atomicité** : Une seule version en stage `prod` à la fois
2. **Immuabilité des artefacts** : Pas de corruption possible depuis MinIO
3. **Audit complet** : Chaque transition enregistrée (timestamp, user, reason)
4. **Rollback instantané** : Reverser une version vers `prod` prend < 1 minute
5. **Vérification d'intégrité** : Les artifacts MinIO ne sont jamais modifiés, seulement lus

---

## Cycle de vie : Entraînement → Promotion → Inférence

### Entraînement (DAG iqa_lifecycle)

```python
# task_train() → MLflow tracking
mlflow.start_run()
mlflow.log_param("learning_rate", 0.001)
mlflow.log_metric("train_loss", 0.05)
mlflow.log_artifact("checkpoint.pt")
run_id = mlflow.active_run().info.run_id  # Récupéré par task_mlflow
```

### Enregistrement (task_mlflow)

```python
def task_mlflow(run_id, scenario_id):
    """Enregistre le run comme version candidate."""

    registered_model_name = f"feature_ae__{scenario_id}"

    # Crée ou récupère le registered model
    # Crée une nouvelle version avec les artefacts du run
    model_version = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=registered_model_name
    )

    # Assign stage: candidate
    mlflow.set_registered_model_alias(
        name=registered_model_name,
        alias="candidate",
        version=model_version.version
    )

    return {
        "registered_model_name": registered_model_name,
        "version": model_version.version,
        "stage": "candidate"
    }
```

### Promotion (task_promotion)

```python
def task_promotion(registered_model_name, version):
    """Promeut de candidate → staging → prod."""

    # candidate → staging
    mlflow.transition_model_version_stage(
        name=registered_model_name,
        version=version,
        stage="staging",
        archive_existing_versions=False
    )

    # [Manual step] staging → prod
    # MLOps valide les métriques, puis:
    mlflow.transition_model_version_stage(
        name=registered_model_name,
        version=version,
        stage="prod",
        archive_existing_versions=True  # Archive l'ancien prod
    )
```

### Inférence (reload)

```python
def reload_model_in_inference_service(scenario_id):
    """Charge le modèle prod pour le scénario."""

    registered_model_name = f"feature_ae__{scenario_id}"

    # Query MLflow (source de vérité)
    versions = mlflow.client.get_latest_versions(
        name=registered_model_name,
        stages=["prod"]
    )

    if versions:
        version = versions[0]
        # POST /admin/reload-model avec artifact_uri
        reload_endpoint(
            artifact_uri=version.source,
            version=version.version
        )
```

---

## Conséquences

- Aucun module applicatif ne peut lire directement un statut "prod" depuis MinIO.
- La promotion = créer ou mettre à jour une version en stage `prod` dans MLflow.
- Le rollback = changer le stage prod vers une version antérieure dans MLflow Registry.
- Les tests d'architecture doivent vérifier que les modèles sont récupérés via MLflow Registry, jamais via une convention S3.
- Les credentials MinIO restent dans `.env` et ne sont jamais commités (ADR 0003).
- **Invariant architecture** : Si MLflow Registry est down, aucun modèle ne peut être promu ou rechargé. MinIO down n'affecte pas les décisions, juste l'accès aux artefacts.
