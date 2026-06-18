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
uv run --extra cpu --extra data dvc pull
uv run --extra cpu --extra data dvc repro
uv run --extra cpu --extra data iqa-check-dvc-reproducibility --with-network true
uv run --extra cpu pytest -q tests/data tests/datasets/test_candidate_builder.py
```

Le remote par defaut reste `iqa-minio`, configure dans `.dvc/config`.

## Airflow

Le DAG `iqa_dvc_reproducibility` expose DVC comme gate de reproductibilite et
de data lineage. Il appelle `iqa-check-dvc-reproducibility` sans reseau par
defaut.

Mode local ou CI Airflow :

```bash
airflow dags trigger iqa_dvc_reproducibility
```

Mode serveur avec MinIO :

```bash
airflow dags trigger iqa_dvc_reproducibility \
  --conf '{"with_network": true}'
```

`--with-network` reste explicite : c'est uniquement dans ce mode que la commande
verifie `dvc pull` et `dvc push` sur `data/raw/hss-iad.dvc`. Les DAGs metier
`iqa_ingestion`, `iqa_replay`, `iqa_monitoring` et `iqa_lifecycle` ne lancent pas
de `dvc push` directement.

DVC est un gate de reproductibilite, pas un declencheur metier. Les replays et
le lifecycle Feature-AE restent declenches par les contrats data et Airflow.

## Validation MinIO

Sur le serveur, charger les variables MinIO avant la verification reseau :

```bash
set -a
source .env
set +a
uv run --extra cpu --extra data dvc remote list
uv run --extra cpu --extra data iqa-check-dvc-reproducibility --with-network true
```

Le script verifie le remote `iqa-minio -> s3://iqa-dvc`, execute `dvc pull`
et `dvc push` sur `data/raw/hss-iad.dvc`, puis regenere les manifests Phase 1/2
et echoue si un diff Git apparait.

## Regles

- Aucun checkpoint `.pt`, masque, heatmap ou image binaire ne doit etre ajoute a Git.
- Les stages doivent produire ou verifier des sorties deterministes.
- Les manifests generes doivent rester compatibles avec `docs/data-contracts.md`.
- Les CSV legers restent Git-tracked dans ce lot ; les `outs` DVC seront ajoutes
  seulement quand l'ownership DVC complet sera valide.
