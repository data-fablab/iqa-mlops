# MLflow Registry - Source de verite

## Decision

MLflow Registry est la source de verite, ou source of truth, pour savoir quelle
version de modele est candidate, en test ou en production. MinIO stocke les
artefacts lourds (`checkpoint.pt`, heatmaps, datasets), mais ne decide pas quel
modele est actif.

References :
- ADR 0006 - MLflow Registry comme source de verite
- ADR 0003 - MinIO comme stockage objet local

## Etat actuel

La branche utilise les aliases MLflow (`candidate`, `test`, `prod`) au lieu des
stages MLflow historiques, car les APIs de stage sont depreciees dans MLflow
recent.

Flux courant :

```text
run MLflow
-> artefact checkpoint sous artifacts/model/checkpoint.pt
-> registered model feature_ae__{scenario_id}
-> version MLflow
-> alias candidate, test ou prod
-> resolution de l'artefact par alias
```

`register_run_to_model()` cree une version de modele depuis l'artefact checkpoint
du run (`.../artifacts/model`) puis positionne l'alias `candidate`.

`promote_model_with_gates()` valide les gates, positionne l'alias cible
(`test` par defaut dans le DAG lifecycle), puis resout l'artefact depuis
MLflow Registry.

`ProdModelLoader` charge uniquement l'alias `prod`.

## Nommage

Un registered model est cree par scenario :

```text
feature_ae__production_replay_natural
feature_ae__drift_domain_extension
```

Format : `<algorithm>__<scenario_id>`.

## Exemple de resolution par alias

```python
import mlflow


def resolve_prod_model(scenario_id: str):
    model_name = f"feature_ae__{scenario_id}"
    client = mlflow.tracking.MlflowClient()
    version = client.get_model_version_by_alias(model_name, "prod")
    return {
        "registered_model_name": model_name,
        "version": version.version,
        "artifact_uri": version.source,
    }
```

## Cycle lifecycle

Mode par defaut du DAG Airflow :

```text
train -> register candidate -> promote alias test -> reload skipped
```

Mode production explicite :

```text
train -> register candidate -> save previous prod -> promote alias prod -> reload prod
```

Le mode production ne doit etre declenche que si `target_stage="prod"` est
fourni au DAG et si les gates passent.

## Invariants

- Aucun module applicatif ne lit un statut de production depuis un prefixe S3.
- La version active est resolue par MLflow Registry et ses aliases.
- Les artefacts MinIO sont passifs et immuables pour une version donnee.
- Si MLflow Registry est indisponible, aucune promotion ni recharge prod ne doit
  etre consideree valide.
