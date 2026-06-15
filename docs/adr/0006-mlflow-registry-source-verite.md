# ADR 0006 - MLflow Registry comme source de verite

## Statut

Accepte.

## Contexte

MinIO stocke les artefacts lourds, mais ne doit pas decider quel modele est actif en production. Une verite parallele de type `s3://iqa-models/prod` creerait un risque d'incoherence avec MLflow.

## Decision

MLflow Registry est la source de verite unique pour la promotion et le rollback.

Un registered model est cree par scenario :

```text
feature_ae__production_replay_natural
feature_ae__drift_domain_extension
```

La promotion et le rollback sont des transitions logiques dans MLflow Registry. MinIO stocke les artefacts, notamment sous `mlflow-artifacts` et `iqa-models`, mais ne porte pas l'etat `prod`.

`/admin/reload-model` resout le modele actif via :

```text
scenario_id -> registered model MLflow -> stage prod -> artifact URI -> reload inference
```

## Consequences

Aucun test ni module runtime ne doit dependre d'un prefixe S3 `prod` comme source de verite. Les artefacts MinIO restent des fichiers, pas une decision de promotion.

## Documentation

Voir [docs/MLflow-Registry.md](../MLflow-Registry.md) pour l'architecture complète, la configuration et les buckets MinIO associés.
