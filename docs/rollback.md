# Rollback et gestion des versions

## Concept

Le **rollback** est la capacité à revenir rapidement à une version antérieure du modèle si la version actuelle (`prod`) montre un problème.

**Principes :**
1. MLflow Registry est la source de vérité unique (ADR 0006)
2. Un rollback = changer le stage `prod` vers une version antérieure dans MLflow
3. Atomicité : MLflow + PostgreSQL metadata store doivent être synchrones
4. Audit complet : chaque transition est enregistrée avec timestamp et raison

---

## États et transitions

### Cycle de vie d'une version

```
create (version v1, v2, v3...)
    ↓
candidate (vient d'être entraînée)
    ↓
staging (gates passed, prêt à passer en prod)
    ↓
prod (actif en production)
    ↓
archived (remplacé ou invalidé)
```

### Diagramme des transitions possibles

```
        ┌─ candidate ─┐
        │             ↓
        │          staging ─┐
        │             ↑      ↓
    [new version] ──┘       prod ──→ archived
        │
        └────────────────────┘
         (direct archive, rarement)
```

### Transitions autorisées

| From | To | Condition | Qui | Quand |
|------|---|-----------|-----|-------|
| `candidate` | `staging` | Gates passed | DAG promotion task | Après éval |
| `staging` | `prod` | Manuel ou scheduled | MLOps / cronjob | Plan de déploiement |
| `prod` | `archived` | Remplacé | MLOps | Après rollback |
| `candidate` | `archived` | Gates failed | Auto-archiving | Rarement |

---

## Scénarios de promotion

### Scenario 1 : Promotion nominale (candidate → staging → prod)

```
1. DAG entraîne version v3
2. task_mlflow : crée feature_ae__production_replay_natural v3 (stage: candidate)
3. task_gates  : valide tous les gates ✓
4. task_promotion : v3 candidate → v3 staging ✓

5. [Manuel] MLOps valide les métriques et logs, décide de passer en prod
6. [Cronjob ou manuel] v3 staging → v3 prod

7. task_reload : charge v3 prod dans l'inference service
   - Query MLflow : registered_model_name + stage prod
   - Récupère artifact_uri
   - POST /admin/reload-model → charge le checkpoint

8. Ancien v2 prod → archived
```

### Scenario 2 : Promotion rapide (candidate → prod, skip staging)

```
1-4. [Même que scenario 1]

5. [Manuel urgent] MLOps décide skip staging (hotfix validé)
6. v3 candidate → v3 prod (direct)
7. task_reload : charge v3 prod

8. v2 prod → archived
```

### Scenario 3 : Rollback après dégradation observée

```
1. v3 prod est actif en production
2. [Observation] Métriques se dégradent : recall < 1.0, AP baisses
3. MLOps décide rollback urgent

4. Query MLflow previous_prod → trouvé v2 prod
5. v3 prod → archived
6. v2 archived → prod (reversal)

7. task_reload : charge v2 prod
   - Mêmes artifacts qu'avant, MLflow est source de vérité
   - Aucune corruption possible (MinIO ne décide pas)

8. Incident créé : "v3 hotfix requis" pour la prochaine itération
```

---

## Métadonnées de transition

### PostgreSQL schema

Table `model_transitions` (ou équivalent metadata store) :

```sql
CREATE TABLE model_transitions (
    id INTEGER PRIMARY KEY,
    registered_model_name VARCHAR,
    version VARCHAR,
    stage_from VARCHAR,  -- "candidate" | "staging" | "prod" | "archived"
    stage_to VARCHAR,
    triggered_by VARCHAR,  -- "dag_task" | "manual" | "rollback_agent" | "auto_archive"
    triggered_at TIMESTAMP,
    reason VARCHAR,  -- "gates_passed" | "mlops_decision" | "incident_response"
    artifact_uri VARCHAR,  -- MinIO S3 URI
    metrics_snapshot JSONB,  -- Sauvegarde des metrics au moment transition
    previous_prod_version VARCHAR,  -- Pour faciliter rollback
    is_rollback BOOLEAN DEFAULT FALSE
);
```

### Exemple transition nominale

```json
{
  "registered_model_name": "feature_ae__production_replay_natural",
  "version": "3",
  "stage_from": "candidate",
  "stage_to": "staging",
  "triggered_by": "dag_task",
  "triggered_at": "2026-06-15T10:30:00Z",
  "reason": "gates_passed",
  "artifact_uri": "s3://mlflow-artifacts/feature_ae__production_replay_natural/3/...",
  "metrics_snapshot": {
    "recall": 1.0,
    "ap": 0.87,
    "latency_ms": 850,
    "orange_rate": 0.08,
    "defect_coverage": 0.95
  },
  "previous_prod_version": "2",
  "is_rollback": false
}
```

---

## Procédure de rollback manuel

### Déclenchement

```bash
# 1. MLOps détecte l'incident
# 2. Valide la version antérieure
mlflow models get-latest-versions \
  --name "feature_ae__production_replay_natural" \
  --stages archived,staging

# 3. Identifie la dernière version stable
version_to_rollback_to=2

# 4. Lance le rollback
mlflow models set-model-version-tag \
  --name "feature_ae__production_replay_natural" \
  --version ${version_to_rollback_to} \
  --key "stage" \
  --value "prod"

# 5. Archive la version défectueuse
mlflow models set-model-version-tag \
  --name "feature_ae__production_replay_natural" \
  --version 3 \
  --key "stage" \
  --value "archived"
```

### Reload automatique

Un cronjob ou webhook MLflow déclenche :

```python
def handle_stage_change_event(event):
    """Déclenché quand une version change de stage dans MLflow."""
    if event["stage"] == "prod":
        # Appelle le reload
        result = reload_model_in_production(
            registered_model_name=event["registered_model_name"],
            scenario_id=extract_scenario_id(event["registered_model_name"])
        )
        # Log la transition
        log_transition(
            event,
            triggered_by="rollback_agent" if event["previous_stage"] == "archived" else "mlops"
        )
```

---

## Artefacts MinIO lors du rollback

### Invariant

**Les artefacts MinIO ne bougent jamais.** Ils restent immuables :

```
s3://mlflow-artifacts/feature_ae__production_replay_natural/3/model/
  └─ checkpoint.pt (fixe, créé au training)
  └─ metadata.json
  └─ hyperparams.yaml

s3://iqa-models/feature_ae__production_replay_natural/3/
  └─ (copie archivée après promotion)
```

**Lors du rollback :**
1. MLflow Registry change : stage prod : v3 → v2
2. Aucune opération MinIO n'est nécessaire
3. task_reload récupère l'artifact_uri pour v2 (qui existe, immuable)
4. Reload l'ancien checkpoint

**Garantie :** Si un artifact a disparu ou était corrompu, le rollback aurait échoué à l'époque où v2 était en prod. L'immuabilité MinIO garantit la reproductibilité.

---

## Déclencheur automatique (Issue 5)

Le rollback **manuel** ci-dessous reste disponible, mais la régression métrique
est désormais détectée et déclenchée automatiquement, en réutilisant le module
`rollback.py` (aucune logique de restauration réécrite). La chaîne reprend
exactement le pattern du drift (règle Prometheus → Alertmanager → sensor) :

```
model-quality-exporter (gauges prod / previous_prod)
  → règle Prometheus IqaModelRegression (deploy/prometheus/rules/iqa_model_regression.rules.yml)
  → Alertmanager (route catch-all) → webhook-catcher
  → iqa_rollback_sensor (lit la série ALERTS via /api/v1/query)
  → iqa_rollback (iqa-run-rollback → rollback_model → reload)
```

- **Gauges prod vs previous_prod** : à chaque promotion prod, `task_promotion`
  appelle `record_prod_promotion_quality(...)` qui rétrograde la run
  `stage=prod` courante en `stage=previous_prod` et logue le nouveau modèle en
  `stage=prod`. L'exporter (Issues 1/2) expose alors les deux séries.
- **Règle de régression** : `IqaModelRegression` tire quand
  `previous_prod − prod > 0.02` sur `pixel_aupimo` (métrique décisive), avec repli
  sur `image_ap` quand les gauges pixel sont absentes (masques GT manquants).
- **Cohérence avec l'Issue 4** : le seuil `0.02` est celui du gate de promotion
  (`configs/promotion_gates.yaml` → `feature_ae.quality_max_regression`) ; un test
  de contrat (`tests/monitoring/test_model_regression_alert.py`) interdit toute
  divergence silencieuse. Même notion de non-régression vs baseline, ici
  `previous_prod` joue la baseline et le prod courant le « candidat ».
- **Sensor** : `iqa_rollback_sensor` (mode `reschedule`, stdlib only, aucun import
  `iqa` dans le scheduler — ADR 0008) lit `ALERTS{alertname="IqaModelRegression"}`
  et déclenche `iqa_rollback`, avec anti-rejeu (pas de second run en vol) et
  cooldown post-rollback.
- **Exécution** : `iqa_rollback` lance `iqa-run-rollback` (résout le prod courant
  comme version fautive, appelle `rollback_model` → restaure `previous_prod`,
  archive la fautive) puis `iqa-run-reload`.

## Alertes et conditions de rollback

### Conditions automatiques (futur)

```yaml
rollback_triggers:
  - metric: recall
    operator: lt
    threshold: 0.99
    duration_minutes: 5  # Alerte si true pendant 5 min

  - metric: latency_p95_ms
    operator: gt
    threshold: 1200
    duration_minutes: 10

  - metric: orange_rate
    operator: gt
    threshold: 0.10
    duration_minutes: 5

  - metric: defect_coverage
    operator: lt
    threshold: 0.90
    duration_minutes: 3
```

### Rollback automatique (optionnel)

```python
def auto_rollback_if_needed():
    """Appelé par monitoring agent toutes les minutes."""
    current_metrics = get_inference_metrics()
    thresholds = load_rollback_triggers()

    if any(exceeds_threshold(m, thresholds) for m in current_metrics.values()):
        # Déclenche rollback auto
        prod_version = get_current_prod_version()
        previous_version = get_previous_prod_version()

        # Log incident
        create_incident(
            title="Auto-rollback triggered",
            description=f"Rolled back from v{prod_version} to v{previous_version}"
        )

        # Bascule MLflow
        set_stage(previous_version, "prod")
        set_stage(prod_version, "archived")

        # Reload
        trigger_reload()
```

---

## Audit et traçabilité

### Logging complet

Chaque transition doit être auditable :

```python
def log_transition(transition_event):
    """Enregistre toute transition de version."""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "registered_model": transition_event["model_name"],
        "version": transition_event["version"],
        "from_stage": transition_event["from"],
        "to_stage": transition_event["to"],
        "triggered_by": transition_event["actor"],  # "dag" | "manual_mlops" | "rollback_agent"
        "reason": transition_event["reason"],
        "user_email": transition_event.get("user"),
        "metrics": transition_event.get("metrics_snapshot"),
    }

    # Log strukturé
    logger.info("model_transition", extra=entry)

    # Stockage PostgreSQL
    db.model_transitions.insert(entry)
```

### Audit trail exemple

```
2026-06-15 10:30:00 | v3 | candidate → staging | dag_task | gates_passed
2026-06-15 14:00:00 | v3 | staging → prod | manual | mlops_review
2026-06-15 16:45:00 | v3 | prod (current, recall=0.98, ap=0.82)
2026-06-15 17:00:00 | v3 | prod → archived | rollback_agent | recall<0.99 for 5min
2026-06-15 17:01:00 | v2 | archived → prod | rollback_agent | auto_rollback
```

---

## FAQ

**Q: Comment savoir quelle version est actuellement en prod ?**
A: Query MLflow : `get_model_version_by_alias(model_name, "prod")`

**Q: Que se passe-t-il si un artifact MinIO disparaît ?**
A: Impossible de loader cette version. Rollback sera bloqué. Audit logs indiqueront la corruption.

**Q: Peut-on avoir plusieurs versions en stage `prod` ?**
A: Non. MLflow enforce : un seul registered model + version en stage `prod` à la fois (immutable stage rule).

**Q: Combien de temps pour un rollback ?**
A: < 1 minute : change stage MLflow + reload inference service.

**Q: Les gates sont-ils re-validées lors du rollback ?**
A: Non. Si v2 était en prod avant, elle a déjà passé les gates. Rollback est une restauration, pas une nouvelle promotion.
