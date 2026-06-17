# DVC Versioning IQA

`dvc.yaml` decrit les stages reproductibles de la couche data Phase 2 sans
mettre les fichiers lourds dans Git. Les CSV legers restent suivis par Git pour
le MVP ; les donnees lourdes et artefacts restent dans MinIO/DVC.

## Stages

| Stage | Role |
| --- | --- |
| `inventory` | Reconstruit l'inventaire images depuis `casting_piece_events.csv` et `data/raw/hss-iad`. |
| `piece_events` | Reapplique les contrats Phase 1/2 et ecrit le rapport de validation data. |
| `replay` | Valide les entrees necessaires aux plans replay naturel et drift. |
| `validation` | Lance les tests de contrats data, no-overlap et metadata Phase 2. |
| `model_dataset` | Verifie le builder de datasets candidats Feature-AE. |

## Commandes

```bash
dvc pull
dvc repro
uv run --extra cpu pytest -q tests/data tests/datasets/test_candidate_builder.py
```

Le remote par defaut reste `iqa-minio`, configure dans `.dvc/config`.

## Regles

- Aucun checkpoint `.pt`, masque, heatmap ou image binaire ne doit etre ajoute a Git.
- Les stages doivent produire ou verifier des sorties deterministes.
- Les manifests generes doivent rester compatibles avec `docs/data-contracts.md`.
