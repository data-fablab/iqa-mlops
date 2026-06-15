# Reproductibilite ML IQA

Cette page decrit le chemin minimal permettant de repartir du dataset source Casting et de produire une prediction avec les modeles retenus.

## Principe

Le repo initial doit livrer le code source reproductible, pas seulement des manifests ou des artefacts deja generes.

Chemin cible :
```text
dataset source Casting
-> ingestion / manifests
-> dataset PyTorch good-only
-> train RD Feature-AE
-> checkpoint Feature-AE
-> predict image
```

Le ROI segmenter et le teacher ResNet18 sont figes. Le RD Feature-AE est le seul modele entrainable dans le MVP.

Les contrats detailles des deux modeles sont separes :

- [Modele Feature-AE IQA](Modele-Feature-AE-IQA.md) ;
- [Modele Segmentation ROI IQA](Modele-Segmentation-ROI-IQA.md).

## Preprocessing Feature-AE

Le preprocessing retenu reprend le principe de la source :

```text
ResizeLetterbox
-> ToTensor
-> Normalize ImageNet
```

Le nom historique `tile_256_overlap` n'est pas conserve dans IQA, car il encode une ancienne taille. Le mode est renomme :

```text
tile_256_overlap -> tiled_context
```

Les tailles sont maintenant des parametres explicites :

```text
preprocessing_mode = tiled_context
tile_size          = 384
context_size       = 768
```

Le mode `letterbox` reste disponible pour un passage image complete simple.

## Responsabilites

```text
src/iqa/ingestion/  -> contrats ingestion historical_replay / production_ingest
src/iqa/datasets/   -> datasets PyTorch depuis manifests CSV
src/iqa/training/   -> train Feature-AE retenu
src/iqa/inference/  -> prediction image/piece_event
src/iqa/models/     -> architectures runtime retenues uniquement
src/iqa/storage/    -> buckets et URI MinIO
scripts/            -> commandes CLI publiques
```

## Commandes

Construire l'inventaire source :
```powershell
uv run --extra cpu iqa-build-inventory --source-dir D:\path\to\hss-iad --output data\metadata\casting_images_inventory.csv
```

Valider que le chemin ML source est present :
```powershell
uv run --extra cpu iqa-validate-ml-source
```

Entrainer un Feature-AE minimal depuis le bootstrap :
```powershell
uv run --extra cpu iqa-train-feature-ae `
  --manifest data\metadata\feature_ae_bootstrap_events.csv `
  --image-root D:\path\to\hss-iad `
  --output-checkpoint models\feature_ae\rd_feature_ae_gated.pt `
  --preprocessing-mode tiled_context `
  --image-size 384 `
  --context-size 768 `
  --epochs 1
```

Predire une image :
```powershell
uv run --extra cpu iqa-predict-image `
  --image D:\path\to\hss-iad\Casting_class1\train\good\sample.jpg `
  --feature-ae-checkpoint models\feature_ae\rd_feature_ae_gated.pt
```

## Stockage

- Dataset source : `s3://iqa-source-datasets` et/ou DVC.
- Images ingerees : `s3://iqa-ingested-images`, avec faits et URI dans PostgreSQL.
- Artefacts : `s3://iqa-models`, `s3://iqa-roi-masks`, `s3://iqa-heatmaps`, `s3://mlflow-artifacts`, `s3://iqa-dvc`.

PostgreSQL stocke les faits et URI. MinIO stocke les fichiers lourds.
Les checkpoints `.pt` ne sont pas versionnes dans Git : Git conserve les
manifests, les versions, les URI MinIO et les checksums ; MinIO conserve les
artefacts lourds comme `s3://iqa-models/roi_segmenter_v001_fixed/checkpoint.pt`.
